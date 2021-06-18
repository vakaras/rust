pub mod aggregate;
pub mod borrowck_errors;
pub mod elaborate_drops;
pub mod patch;
pub mod storage;

pub mod alignment;
pub mod collect_writes;
pub mod find_self_call;
pub mod generic_graph;
pub mod generic_graphviz;
pub mod graphviz;
pub mod pretty;
pub mod spanview;

pub use self::aggregate::expand_aggregate;
pub use self::alignment::is_disaligned;
pub use self::find_self_call::find_self_call;
pub use self::generic_graph::graphviz_safe_def_name;
pub use self::graphviz::write_mir_graphviz;
pub use self::pretty::{dump_enabled, dump_mir, write_mir_pretty, PassWhere};
