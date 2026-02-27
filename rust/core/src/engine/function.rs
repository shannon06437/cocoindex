use crate::engine::context::{
    ComponentProcessorContext, FnCallContext, FnCallMemo, FnCallMemoEntry,
};
use crate::engine::execution::read_fn_call_memo;
use crate::engine::profile::EngineProfile;
use crate::prelude::*;

use cocoindex_utils::fingerprint::Fingerprint;
use tokio::sync::OwnedRwLockReadGuard;

pub struct PendingFnCallMemo<Prof: EngineProfile> {
    // `FnCallMemoEntry` expected to be in Pending state.
    guard: tokio::sync::OwnedRwLockWriteGuard<FnCallMemoEntry<Prof>>,
}

impl<Prof: EngineProfile> PendingFnCallMemo<Prof> {
    pub fn resolve(
        mut self,
        fn_ctx: &FnCallContext,
        ret: impl FnOnce() -> Prof::FunctionData,
    ) -> Result<bool> {
        let has_child_components = fn_ctx.update(|inner| inner.has_child_components);
        if has_child_components {
            *self.guard = FnCallMemoEntry::Ready(None);
            client_bail!(
                "A function with memo=True mounted child components. \
                 Either mount the function as a component, or set memo=False."
            );
        }
        let memo_ret = fn_ctx.update(|inner| {
            Some(FnCallMemo {
                ret: ret(),
                target_state_paths: std::mem::take(&mut inner.target_state_paths),
                dependency_memo_entries: std::mem::take(&mut inner.dependency_memo_entries),
                logic_deps: inner.logic_deps.clone(),
                already_stored: false,
            })
        });
        let resolved = memo_ret.is_some();
        *self.guard = FnCallMemoEntry::Ready(memo_ret);
        Ok(resolved)
    }
}

pub enum FnCallMemoGuard<Prof: EngineProfile> {
    Ready(tokio::sync::OwnedRwLockReadGuard<FnCallMemoEntry<Prof>, Option<FnCallMemo<Prof>>>),
    Pending(PendingFnCallMemo<Prof>),
}

pub(crate) async fn reserve_memoization_with_reader<Prof: EngineProfile>(
    comp_exec_ctx: &ComponentProcessorContext<Prof>,
    memo_fp: Fingerprint,
    memo_reader: impl Fn(
        &ComponentProcessorContext<Prof>,
        Fingerprint,
    ) -> Result<Option<FnCallMemo<Prof>>>,
) -> Result<FnCallMemoGuard<Prof>> {
    // println!("BEGIN reserve_memoization in Rust");
    let mut try_write = false;
    loop {
        // We clone out the Arc so we don't hold any mutexes across `.await`.
        let memo_entry =
            comp_exec_ctx.update_building_state(|building_state| {
                match building_state.fn_call_memos.entry(memo_fp) {
                    std::collections::hash_map::Entry::Occupied(e) => Ok(e.get().clone()),
                    std::collections::hash_map::Entry::Vacant(e) => {
                        try_write = true;
                        let entry = Arc::new(tokio::sync::RwLock::new(FnCallMemoEntry::Pending));
                        e.insert(entry.clone());
                        Ok(entry)
                    }
                }
            })?;

        let result = if try_write {
            // If pending, attempt to become the resolver by acquiring a write lock.
            let mut guard = memo_entry.write_owned().await;
            if let FnCallMemoEntry::Pending = &*guard {
                // Under full_reprocess, force execution (do not load cached memo).
                if !comp_exec_ctx.full_reprocess() {
                    let stored_fn_call_memo = memo_reader(comp_exec_ctx, memo_fp)?;
                    if let Some(fn_call_memo) = stored_fn_call_memo {
                        *guard = FnCallMemoEntry::Ready(Some(fn_call_memo));
                    }
                }
            }
            match &mut *guard {
                FnCallMemoEntry::Ready(_) => {
                    let ready_guard =
                        tokio::sync::OwnedRwLockReadGuard::map(guard.downgrade(), |mem_entry| {
                            match mem_entry {
                                FnCallMemoEntry::Ready(memo) => memo,
                                _ => unreachable!(),
                            }
                        });
                    FnCallMemoGuard::Ready(ready_guard)
                }
                FnCallMemoEntry::Pending => FnCallMemoGuard::Pending(PendingFnCallMemo { guard }),
            }
        } else {
            let read_guard = memo_entry.read_owned().await;
            let ready_guard =
                OwnedRwLockReadGuard::try_map(read_guard, |mem_entry| match mem_entry {
                    FnCallMemoEntry::Ready(memo) => Some(memo),
                    _ => None,
                });
            match ready_guard {
                Ok(ready_guard) => FnCallMemoGuard::Ready(ready_guard),
                Err(_) => {
                    // Edge case: The initial call that creates the pending entry doesn't finish, e.g. it can be an exception.
                    // We need to read the entry from the map again and try to grab the write lock.
                    try_write = true;
                    continue;
                }
            }
        };
        return Ok(result);
    }
}

pub async fn reserve_memoization<Prof: EngineProfile>(
    comp_exec_ctx: &ComponentProcessorContext<Prof>,
    memo_fp: Fingerprint,
) -> Result<FnCallMemoGuard<Prof>> {
    reserve_memoization_with_reader(comp_exec_ctx, memo_fp, read_fn_call_memo).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::component::{Component, ComponentProcessor, ComponentProcessorInfo};
    use crate::engine::context::{AppContext, ComponentProcessingMode};
    use crate::engine::environment::{AppRegistration, Environment, EnvironmentSettings};
    use crate::engine::stats::ProcessingStats;
    use crate::engine::target_state::{
        ChildTargetDef, TargetActionSink, TargetHandler, TargetReconcileOutput,
        TargetStateProviderRegistry,
    };
    use crate::state::stable_path::{StableKey, StablePath};
    // Minimal EngineProfile for testing (no real persistence needed).
    #[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
    struct TestProf;

    #[derive(Clone)]
    struct TestComponentProc;

    impl ComponentProcessor<TestProf> for TestComponentProc {
        fn process(
            &self,
            _host_runtime_ctx: &(),
            _comp_ctx: &ComponentProcessorContext<TestProf>,
        ) -> Result<impl Future<Output = Result<Vec<u8>>> + Send + 'static> {
            Ok(async { Ok(vec![]) })
        }
        fn memo_key_fingerprint(&self) -> Option<Fingerprint> {
            None
        }
        fn processor_info(&self) -> &ComponentProcessorInfo {
            static INFO: OnceLock<ComponentProcessorInfo> = OnceLock::new();
            INFO.get_or_init(|| ComponentProcessorInfo::new("test".to_string()))
        }
    }

    #[derive(Debug, Clone, PartialEq, Eq, Hash)]
    struct TestSink;

    #[async_trait]
    impl TargetActionSink<TestProf> for TestSink {
        async fn apply(
            &self,
            _host_runtime_ctx: &(),
            _actions: Vec<()>,
        ) -> Result<Option<Vec<Option<ChildTargetDef<TestProf>>>>> {
            Ok(None)
        }
    }

    struct TestHandler;

    impl TargetHandler<TestProf> for TestHandler {
        fn reconcile(
            &self,
            _key: StableKey,
            _desired: Option<()>,
            _prev: &[Vec<u8>],
            _prev_may_be_missing: bool,
        ) -> Result<Option<TargetReconcileOutput<TestProf>>> {
            Ok(None)
        }
    }

    impl crate::engine::profile::Persist for Vec<u8> {
        fn to_bytes(&self) -> Result<bytes::Bytes> {
            Ok(bytes::Bytes::from(self.clone()))
        }
        fn from_bytes(data: &[u8]) -> Result<Self> {
            Ok(data.to_vec())
        }
    }

    impl crate::engine::profile::EngineProfile for TestProf {
        type HostRuntimeCtx = ();
        type ComponentProc = TestComponentProc;
        type FunctionData = Vec<u8>;
        type TargetHdl = TestHandler;
        type TargetStateTrackingRecord = Vec<u8>;
        type TargetAction = ();
        type TargetActionSink = TestSink;
        type TargetStateValue = ();
    }

    struct TestCtx {
        _tmp: tempfile::TempDir,
        ctx: ComponentProcessorContext<TestProf>,
    }

    fn build_test_ctx(full_reprocess: bool) -> TestCtx {
        let tmp_dir = tempfile::tempdir().expect("create temp dir");
        let settings = EnvironmentSettings {
            db_path: tmp_dir.path().to_path_buf(),
        };
        let providers_reg = TargetStateProviderRegistry::new(rpds::HashTrieMapSync::new_sync());
        let env = Environment::<TestProf>::new(settings, Arc::new(Mutex::new(providers_reg)), ())
            .expect("create test environment");

        let app_reg = AppRegistration::new("test_app", &env).expect("register app");
        let db = {
            let mut wtxn = env.db_env().write_txn().expect("write txn");
            let db = env
                .db_env()
                .create_database(&mut wtxn, Some("test_app"))
                .expect("create db");
            wtxn.commit().expect("commit");
            db
        };
        let app_ctx = AppContext::new(env, db, app_reg, None);
        let component = Component::<TestProf>::new(app_ctx, StablePath::root());
        let providers = rpds::HashTrieMapSync::new_sync();
        let stats = ProcessingStats::default();
        let ctx = ComponentProcessorContext::new(
            component,
            providers,
            None,
            stats,
            ComponentProcessingMode::Build,
            full_reprocess,
        );

        TestCtx { _tmp: tmp_dir, ctx }
    }

    #[tokio::test]
    async fn full_reprocess_does_not_read_cached_memo() {
        let test_ctx = build_test_ctx(true);
        let memo_fp = Fingerprint::from_bytes(&[0u8; 16]);

        let guard = reserve_memoization_with_reader(&test_ctx.ctx, memo_fp, |_ctx, _fp| {
            panic!("memo reader should NOT be called when full_reprocess=true");
        })
        .await
        .expect("reserve_memoization_with_reader should succeed");

        assert!(
            matches!(guard, FnCallMemoGuard::Pending(_)),
            "Under full_reprocess, reserve_memoization should return Pending (not Ready)"
        );
    }

    #[tokio::test]
    async fn normal_mode_does_read_cached_memo() {
        let test_ctx = build_test_ctx(false);
        let memo_fp = Fingerprint::from_bytes(&[0u8; 16]);

        let guard = reserve_memoization_with_reader(&test_ctx.ctx, memo_fp, |_ctx, _fp| Ok(None))
            .await
            .expect("reserve_memoization_with_reader should succeed");

        // With no stored memo and full_reprocess=false, it should still be Pending
        // (because the reader returned None).
        assert!(
            matches!(guard, FnCallMemoGuard::Pending(_)),
            "With no stored memo, should return Pending"
        );
    }
}
