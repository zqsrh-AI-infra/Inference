pub mod routes;
pub mod handlers;
pub mod models;
pub mod inference;
pub mod chat;
pub mod embeddings;
pub mod rerank;

pub use routes::create_router;