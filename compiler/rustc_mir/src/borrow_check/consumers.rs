//! This file provides API for compiler consumers.

use super::{
    borrow_set::BorrowSet, dataflow::Analysis, facts::AllFacts, gather_closure_upvars,
    nll
};
use crate::dataflow::impls::MaybeInitializedPlaces;
use crate::dataflow::move_paths::MoveData;
use crate::dataflow::MoveDataParamEnv;
use rustc_index::vec::IndexVec;
use rustc_infer::infer::{InferCtxt, TyCtxtInferExt};
use rustc_middle::mir::{Body, Promoted};
use rustc_middle::ty::TyCtxt;
use std::rc::Rc;

pub use super::{
    facts::{AllFacts as PoloniusInput, RustcFacts},
    nll::PoloniusOutput, location::{LocationTable, RichLocation},
};

/// A `Body` with Polonius facts. We include the body here because the region
/// identifiers must match the ones in the Polonius facts.
pub struct BodyWithFacts<'tcx> {
    /// A mir body that contains region identifiers.
    pub body: Body<'tcx>,
    /// Polonius input facts.
    pub input_facts: AllFacts,
    /// Polonius output facts.
    pub output_facts: PoloniusOutput,
    /// The table that maps Polonius points to locations in the table.
    pub location_table: LocationTable,
}

/// This function computes Polonius facts for the given body. It makes a copy of
/// the body because it needs to regenerate the region identifiers.
pub fn compute_polonius_facts<'tcx>(
    tcx: TyCtxt<'tcx>,
    input_body: &Body<'tcx>,
    promoted: IndexVec<Promoted, Body<'tcx>>,
) -> BodyWithFacts<'tcx> {
    let mut result = None;
    let result_borrow = &mut result;
    tcx.infer_ctxt().enter(|infcx| {
        *result_borrow = Some(do_compute_polonius_facts(&infcx, input_body, promoted));
    });
    result.unwrap()
}

/// The implementation of this function is based on `do_mir_borrowck`.
fn do_compute_polonius_facts<'tcx>(
    infcx: &InferCtxt<'_, 'tcx>,
    input_body: &Body<'tcx>,
    mut promoted: IndexVec<Promoted, Body<'tcx>>,
) -> BodyWithFacts<'tcx> {
    let def = input_body.source.with_opt_param().as_local().unwrap();

    debug!("do_compute_polonius_facts(def = {:?})", def);

    let tcx = infcx.tcx;
    let param_env = tcx.param_env(def.did);
    let id = tcx.hir().local_def_id_to_hir_id(def.did);

    let upvars = gather_closure_upvars(infcx, def);

    // Replace all regions with fresh inference variables. This
    // requires first making our own copy of the MIR. This copy will
    // be modified (in place) to contain non-lexical lifetimes. It
    // will have a lifetime tied to the inference context.
    let mut body = input_body.clone();
    let free_regions = nll::replace_regions_in_mir(infcx, param_env, &mut body, &mut promoted);

    let location_table = LocationTable::new(&body);

    let move_data = MoveData::gather_moves(&body, tcx, param_env)
        .expect("typechecked body should not have move errors");
    let mdpe = MoveDataParamEnv { move_data, param_env };

    let mut flow_inits = MaybeInitializedPlaces::new(tcx, &body, &mdpe)
        .into_engine(tcx, &body)
        .pass_name("borrowck")
        .iterate_to_fixpoint()
        .into_results_cursor(&body);

    let locals_are_invalidated_at_exit = tcx.hir().body_owner_kind(id).is_fn_or_closure();
    let borrow_set =
        Rc::new(BorrowSet::build(tcx, &body, locals_are_invalidated_at_exit, &mdpe.move_data));

    let nll::NllOutput {
        regioncx: _,
        opaque_type_values: _,
        polonius_input,
        polonius_output,
        opt_closure_req: _,
        nll_errors,
    } = nll::compute_regions(
        infcx,
        free_regions,
        &body,
        &promoted,
        &location_table,
        param_env,
        &mut flow_inits,
        &mdpe.move_data,
        &borrow_set,
        &upvars,
    );
    assert!(nll_errors.is_empty());

    let input_facts = *polonius_input.expect("Polonius input facts were not generated");
    let output_facts =
        Rc::try_unwrap(polonius_output.expect("Polonius output was not computed")).unwrap();

    BodyWithFacts { body, input_facts, output_facts, location_table }
}
