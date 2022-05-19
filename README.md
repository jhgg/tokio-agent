"Agents are a simple abstraction around state."

This crate is inspired by Elixir's [Agent](https://hexdocs.pm/elixir/Agent.html).

In Rust however, an Agent holds your State in either a tokio task, or a thread, and allows you to interact with that
state, while having the state remain within the task or thread.

An agent managing state in a thread for examplue could be used to manage and interact with a resource which would
cause blocking on the tokio runtime (for example, sqlite database).

An agent managing state in a task could be used to manage and interact with a resource in a lock-free way.
