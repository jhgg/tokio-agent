#![warn(missing_docs)]

//! "Agents are a simple abstraction around state."
//!
//! This crate is inspired by Elixir's [Agent](https://hexdocs.pm/elixir/Agent.html).
//!
//! In Rust however, an Agent holds your State in either a tokio task, or a thread, and allows you to interact with that
//! state, while having the state remain within the task or thread.
//!
//! An agent managing state in a thread for examplue could be used to manage and interact with a resource which would
//! cause blocking on the tokio runtime (for example, sqlite database).
//!
//! An agent managing state in a task could be used to manage and interact with a resource in a lock-free way.

use std::any::type_name;

use tokio::sync::{
    mpsc::{unbounded_channel, UnboundedSender},
    oneshot,
};

/// A convenience wrapper result type, for use when dealing with the [`Handle`] and [`BlockingHandle`] API.
pub type Result<T, E = Stopped> = std::result::Result<T, E>;

/// See the crate's main documentation for a description of what an [`Agent`] is.
///
/// The [`Agent`] is also an enum that can be used to control the [`Agent`] when using [`Handle::call`].
pub enum Agent<T = ()> {
    /// Stop the agent.
    Stop(T),
    /// Continue running the agent.
    Continue(T),
}

impl<T> Agent<T> {
    /// Spawns an agent that will manage the state returned by the `initial_state` function in its own thread.
    ///
    /// This function is identical to [`Agent::spawn_thread`], except that it allows for you to name the thread.
    pub fn spawn_thread_named<F>(initial_state: F, name: String) -> std::io::Result<Handle<T>>
    where
        F: FnOnce() -> T + Send + 'static,
        T: 'static,
    {
        let (sender, mut receiver) = unbounded_channel::<Evaluate<T>>();

        std::thread::Builder::new().name(name).spawn(move || {
            let mut state = initial_state();

            while let Some(evaluator) = receiver.blocking_recv() {
                match (evaluator)(&mut state) {
                    Agent::Stop(_) => break,
                    Agent::Continue(_) => continue,
                }
            }
        })?;

        Ok(Handle::new(sender))
    }

    /// Spawns an agent that will manage the state returned by `initial_state` in its own thread.
    ///
    /// The thread will be named after the type returned by `initial_state`.
    pub fn spawn_thread<F>(initial_state: F) -> std::io::Result<Handle<T>>
    where
        F: FnOnce() -> T + Send + 'static,
        T: 'static,
    {
        Self::spawn_thread_named(initial_state, format!("Agent({})", type_name::<T>()))
    }

    /// Spawns an agent that will manage the state returned by `initial_state` a task using [tokio::task::spawn].
    ///
    /// This function must be called from the context of a Tokio runtime.
    ///
    /// # Panics
    ///
    /// Panics if called from **outside** of the Tokio runtime.
    pub fn spawn_task<F>(initial_state: F) -> Handle<T>
    where
        F: FnOnce() -> T + Send + 'static,
        T: Send + 'static,
    {
        let (sender, mut receiver) = unbounded_channel::<Evaluate<T>>();

        tokio::task::spawn(async move {
            let mut state = initial_state();

            while let Some(evaluator) = receiver.recv().await {
                match (evaluator)(&mut state) {
                    Agent::Stop(_) => break,
                    Agent::Continue(_) => continue,
                }
            }
        });

        Handle::new(sender)
    }

    /// Spawns an agent that will manage the state returned by `initial_state` in a local task using [tokio::task::spawn_local].
    ///
    /// The agent will be run on the same thread that called `spawn_local.` This may only be called from the context
    /// of a local task set.
    ///
    /// # Panics
    ///
    /// Panics if called **outside** of a *local task set*.
    pub fn spawn_local_task<F>(initial_state: F) -> Handle<T>
    where
        F: FnOnce() -> T + 'static,
        T: 'static,
    {
        let (sender, mut receiver) = unbounded_channel::<Evaluate<T>>();

        tokio::task::spawn_local(async move {
            let mut state = initial_state();

            while let Some(evaluator) = receiver.recv().await {
                match (evaluator)(&mut state) {
                    Agent::Stop(_) => break,
                    Agent::Continue(_) => continue,
                }
            }
        });

        Handle::new(sender)
    }

    #[inline(always)]
    fn transpose(self) -> (T, Agent<()>) {
        match self {
            Agent::Stop(inner) => (inner, Agent::Stop(())),
            Agent::Continue(inner) => (inner, Agent::Continue(())),
        }
    }
}

type Evaluate<T> = Box<dyn FnOnce(&mut T) -> Agent<()> + Send + 'static>;

/// The agent that this handle refers to has already been stopped.
#[derive(Debug)]
pub struct Stopped;

/// A [`Handle`] allows for interaction with a previously spawned [`Agent`] inside of a tokio runtime context.
///
/// If you wish to interact with the [`Agent`] outside of a tokio runtime context, use [`Handle::blocking`], which
/// will turn this handle into a [`BlockingHandle`].
///
/// The [`Agent`] will be automatically stopped once all [`Handle`] and [`BlockingHandle`] that refer to the
/// agent are dropped, or if the `stop` method is called.
///
/// See the function implementations on this struct for more information on what you can do with an [`Agent`].
#[derive(Clone)]
pub struct Handle<T> {
    common: HandleCommon<T>,
}

impl<T> std::ops::Deref for Handle<T> {
    type Target = HandleCommon<T>;

    fn deref(&self) -> &Self::Target {
        &self.common
    }
}

/// Contains common functionality that is shared between [`Handle`] and [`BlockingHandle`].
#[derive(Clone)]
pub struct HandleCommon<T> {
    sender: UnboundedSender<Evaluate<T>>,
}

impl<T> HandleCommon<T> {
    #[inline(always)]
    fn do_call<F, R>(&self, func: F) -> Result<oneshot::Receiver<R>>
    where
        F: FnOnce(&mut T) -> Agent<R> + Send + 'static,
        R: Send + 'static,
    {
        let (sender, receiver) = oneshot::channel();

        self.sender
            .send(Box::new(move |state| {
                let (result, agent) = func(state).transpose();
                sender.send(result).ok();
                agent
            }))
            .map_err(|_| Stopped)?;

        Ok(receiver)
    }

    /// Checks if the agent has been stopped.
    ///
    /// ```
    /// use tokio_agent::{Agent, Result};
    ///
    /// #[tokio::main]
    /// async fn main() -> Result<()> {
    ///     let agent = Agent::spawn_thread(|| 1).unwrap();
    ///     let agent_2 = agent.clone();
    ///
    ///     agent.stop().await?;
    ///
    ///     assert!(agent_2.is_stopped());
    ///
    ///     Ok(())
    /// }
    /// ```
    #[inline(always)]
    pub fn is_stopped(&self) -> bool {
        self.sender.is_closed()
    }

    /// Performs a cast (*fire and forget*) operation on the agent state.
    ///
    /// The function `func` is sent to the agent, which then invokes the function, passing a
    /// mutable reference to the agent's state.
    ///
    /// This function will return true if the agent was not stopped at the time of invoking `cast`,
    /// however, a true result does not guarantee that your function will actually exit, as the agent may be stopped
    /// prior to evaluating your function.
    ///
    /// ```
    /// use tokio_agent::Agent;
    ///
    /// let agent = Agent::spawn_thread(|| 42).unwrap().blocking();
    ///
    /// agent.cast(|x| {
    ///     *x += 1;
    ///     Agent::Continue(())
    /// });
    /// assert_eq!(agent.get(|x| *x).unwrap(), 43);
    /// ```
    #[inline(always)]
    pub fn cast<F>(&self, func: F) -> bool
    where
        F: FnOnce(&mut T) -> Agent<()> + Send + 'static,
    {
        self.sender.send(Box::new(move |state| func(state))).is_ok()
    }

    /// Returns `true` if the [`Agent`] refered to by this handle is the same as `other`'s Agent.
    ///
    /// ```
    /// use tokio_agent::Agent;
    ///
    /// let agent = Agent::spawn_thread(|| "the agent").unwrap();
    /// let agent_2 = agent.clone();
    ///
    /// assert!(agent_2.is_same_agent(&agent));
    ///
    /// // Additionally, a BlockingHandle can be compared with an Handle as well.
    ///
    /// let agent_2 = agent_2.blocking();
    /// assert!(agent_2.is_same_agent(&agent));
    /// assert!(agent.is_same_agent(&agent_2));
    ///
    /// let imposter = Agent::spawn_thread(|| "sus").unwrap();
    /// assert!(!agent.is_same_agent(&imposter));
    ///
    ///
    #[inline(always)]
    pub fn is_same_agent(&self, other: &HandleCommon<T>) -> bool {
        self.sender.same_channel(&other.sender)
    }
}

impl<T> std::fmt::Debug for Handle<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("Handle").field(&type_name::<T>()).finish()
    }
}

/// The "Blocking" variant of [`Handle`] which can be used to interact with an [`Agent`] outside of a tokio runtime
/// context.
#[derive(Clone)]
pub struct BlockingHandle<T>(Handle<T>);

impl<T> std::fmt::Debug for BlockingHandle<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("BlockingHandle")
            .field(&type_name::<T>())
            .finish()
    }
}

impl<T> std::ops::Deref for BlockingHandle<T> {
    type Target = HandleCommon<T>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> Handle<T> {
    fn new(sender: UnboundedSender<Evaluate<T>>) -> Self {
        Self {
            common: HandleCommon { sender },
        }
    }

    /// Converts this [`Handle`] into a [`BlockingHandle`] which allows you to interact with the the [`Agent`]
    /// outside of a tokio runtime context.
    pub fn blocking(self) -> BlockingHandle<T> {
        BlockingHandle(self)
    }

    /// Makes a synchronous call to the agent, waiting for the agent to evaluate the provided function.
    ///
    /// The function `func` is sent to the agent, which then invokes the function, passing a
    /// mutable reference to the agent's state. The return value of `func` is then sent back to the current
    /// task.
    ///
    /// If the agent is stopped before evaluating your function, an error will be returned.
    ///
    /// Unless you need to dynamically decide whether the agent should stop or continue, `get`, `get_and_update`, `update`,
    /// `replace`, `take` and `stop` can be used instead to avoid a bit of boilerplate in specifying whether the agent
    /// should stop or continue.
    ///
    /// ```
    /// use tokio_agent::{Agent, Result};
    ///
    /// #[tokio::main]
    /// async fn main() -> Result<()> {
    ///     let agent = Agent::spawn_task(|| 0);
    ///
    ///     let result = agent.call(|x| {
    ///         *x += 1;
    ///         Agent::Continue(*x)
    ///     }).await?;
    ///
    ///     assert_eq!(result, 1);
    ///     Ok(())
    /// }
    pub async fn call<F, R>(&self, func: F) -> Result<R>
    where
        F: FnOnce(&mut T) -> Agent<R> + Send + 'static,
        R: Send + 'static,
    {
        Ok(self.do_call(func)?.await.map_err(|_| Stopped)?)
    }

    /// Makes a synchronous call to the agent, asking it to clone its state and send it back to you.
    ///
    /// If the agent is stopped before the state is able to be cloned, an error will be returned.
    ///
    /// ```
    /// use tokio_agent::{Agent, Result};
    ///
    /// #[tokio::main]
    /// async fn main() -> Result<()> {
    ///     let agent = Agent::spawn_task(|| 42);
    ///     let result = agent.clone_state().await?;
    ///     assert_eq!(result, 42);
    ///     Ok(())
    /// }
    /// ```
    pub async fn clone_state(&self) -> Result<T>
    where
        T: Send + Clone + 'static,
    {
        self.call(|state| Agent::Continue(state.clone())).await
    }

    /// Makes a synchronous call to the agent, invoking the provided function with an *immutable* reference
    /// to the [`Agent`]'s state, and returning the value returned from the function to you.
    ///
    /// If the agent is stopped before the state is able to be cloned, an error will be returned.
    ///
    /// ```
    /// use tokio_agent::{Agent, Result};
    ///
    /// #[tokio::main]
    /// async fn main() -> Result<()> {
    ///     let agent = Agent::spawn_task(|| 42);
    ///     let result = agent.get(|x| *x).await?;
    ///     assert_eq!(result, 42);
    ///     Ok(())
    /// }
    /// ```
    pub async fn get<F, R>(&self, func: F) -> Result<R>
    where
        F: FnOnce(&T) -> R + Send + 'static,
        R: Send + 'static,
    {
        self.call(move |state| Agent::Continue(func(state))).await
    }

    /// Makes a synchronous call to the agent, invoking the provided function with a _mutable_ reference
    /// to the [`Agent`]'s state, and returning the value returned from the function to you.
    ///
    /// If the agent is stopped before the provided function is able to be evaluated, an error will be returned.
    ///
    /// ```
    /// use tokio_agent::{Agent, Result};
    ///
    /// #[tokio::main]
    /// async fn main() -> Result<()> {
    ///     let agent = Agent::spawn_task(|| 42);
    ///     let result = agent.get_and_update(|x| {
    ///         *x += 1;
    ///         *x
    ///     }).await?;
    ///     assert_eq!(result, 43);
    ///
    ///     Ok(())
    /// }
    /// ```
    pub async fn get_and_update<F, R>(&self, func: F) -> Result<R>
    where
        F: FnOnce(&mut T) -> R + Send + 'static,
        R: Send + 'static,
    {
        self.call(move |state| Agent::Continue(func(state))).await
    }

    /// Makes a synchronous call to the agent, invoking the provided function with a _mutable_ reference
    /// to the [`Agent`]'s state.
    ///
    /// If the agent is stopped before the function is able to be evaluated, an error will be returned.
    ///
    /// ```
    /// use tokio_agent::{Agent, Result};
    ///
    /// #[tokio::main]
    /// async fn main() -> Result<()> {
    ///     let agent = Agent::spawn_task(|| 42);
    ///     agent.update(|x| { *x += 1; }).await?;
    ///     let result = agent.get(|x| *x).await?;
    ///     assert_eq!(result, 43);
    ///     Ok(())
    /// }
    pub async fn update<F>(&self, func: F) -> Result<()>
    where
        F: FnOnce(&mut T) + Send + 'static,
    {
        self.call(|x| Agent::Continue(func(x))).await
    }

    /// Makes a synchronous call to the agent, asking the agent to replace its state with `new_state`, returning the
    /// old state of the agent.
    ///
    /// If the agent is stopped before the state is able to be replaced, an error will be returned.
    ///
    /// ```
    /// use tokio_agent::{Agent, Result};
    ///
    /// #[tokio::main]
    /// async fn main() -> Result<()> {
    ///     let agent = Agent::spawn_task(|| 42);
    ///     let result = agent.replace(43).await?;
    ///     assert_eq!(result, 42);
    ///     let result = agent.get(|x| *x).await?;
    ///     assert_eq!(result, 43);
    ///     Ok(())
    /// }
    /// ```
    pub async fn replace(&self, new_state: T) -> Result<T>
    where
        T: Send + 'static,
    {
        self.get_and_update(|state| std::mem::replace(state, new_state))
            .await
    }

    /// Makes a synchronous call to the agent, asking the agent to replace its state with a new state, constructed using
    /// [`Default::default`]. The old state is returned back to you.
    ///
    /// If the agent is stopped before the state is able to be taken, an error will be returned.
    ///
    /// ```
    /// use tokio_agent::{Agent, Result};
    ///
    /// #[tokio::main]
    /// async fn main() -> Result<()> {
    ///     let agent = Agent::spawn_task(|| 42);
    ///     let result = agent.take().await?;
    ///     assert_eq!(result, 42);
    ///     let result = agent.get(|x| *x).await?;
    ///     assert_eq!(result, 0);
    ///
    ///     Ok(())
    /// }
    /// ```
    pub async fn take(&self) -> Result<T>
    where
        T: Default + Send + 'static,
    {
        self.get_and_update(|state| std::mem::take(state)).await
    }

    /// Makes a synchronous call to the agent, asking the agent to stop.
    ///
    /// Returns when the agent has successfully stopped.
    ///
    /// If the agent was stopped prior to this function was invoked, an error will be returned.
    ///
    /// ```
    /// use tokio_agent::{Agent, Result};
    ///
    /// #[tokio::main]
    /// async fn main() -> Result<()> {
    ///     let agent = Agent::spawn_task(|| 42);
    ///     let agent_2 = agent.clone();
    ///
    ///     agent.stop().await?;
    ///
    ///     assert!(agent_2.stop().await.is_err());
    ///
    ///     Ok(())
    /// }
    /// ```
    pub async fn stop(self) -> Result<()> {
        self.call(|_| Agent::Stop(())).await
    }

    /// Monitors the [`Agent`] refered to by this handle, returning when the agent dies.
    ///
    /// This function is cancel-safe.
    ///
    /// ```
    /// use tokio_agent::Agent;
    ///
    /// #[tokio::main]
    /// async fn main() {
    ///     let agent = Agent::spawn_task(|| 42);
    ///     let agent_2 = agent.clone();
    ///
    ///     tokio::task::spawn(async move {
    ///         agent_2.stop().await.unwrap();
    ///     });
    ///
    ///     agent.monitor().await
    /// }
    /// ```
    pub async fn monitor(&self) {
        self.sender.closed().await
    }
}

impl<T> BlockingHandle<T> {
    /// Converts this [`BlockingHandle`] back into an [`Handle`] which allows you to interact with the the
    /// [`Agent`] inside of a tokio runtime context.
    pub fn nonblocking(self) -> Handle<T> {
        self.0
    }

    /// The blocking variant of [`Handle::call`].
    ///
    /// ```
    /// use tokio_agent::{Agent, Result};
    ///
    /// fn main() -> Result<()> {
    ///     let agent = Agent::spawn_thread(|| 0).unwrap().blocking();
    ///     let result = agent.call(|x| {
    ///         *x += 1;
    ///         Agent::Continue(*x)
    ///     })?;
    ///
    ///     assert_eq!(result, 1);
    ///     Ok(())
    /// }
    /// ```
    pub fn call<F, R>(&self, func: F) -> Result<R>
    where
        F: FnOnce(&mut T) -> Agent<R> + Send + 'static,
        R: Send + 'static,
    {
        Ok(self.do_call(func)?.blocking_recv().map_err(|_| Stopped)?)
    }

    /// The blocking variant of [`Handle::clone_state`].
    ///
    /// ```
    /// use tokio_agent::{Agent, Result};
    ///
    /// fn main() -> Result<()> {
    ///     let agent = Agent::spawn_thread(|| 42).unwrap().blocking();
    ///     let result = agent.clone_state()?;
    ///     assert_eq!(result, 42);
    ///     Ok(())
    /// }
    /// ```
    pub fn clone_state(&self) -> Result<T>
    where
        T: Send + Clone + 'static,
    {
        self.call(|state| Agent::Continue(state.clone()))
    }

    /// The blocking variant of [`Handle::get`].
    ///
    /// ```
    /// use tokio_agent::{Agent, Result};
    ///
    /// fn main() -> Result<()> {
    ///     let agent = Agent::spawn_thread(|| 42).unwrap().blocking();
    ///     let result = agent.get(|x| *x)?;
    ///     assert_eq!(result, 42);
    ///     Ok(())
    /// }
    /// ```
    pub fn get<F, R>(&self, func: F) -> Result<R>
    where
        F: FnOnce(&T) -> R + Send + 'static,
        R: Send + 'static,
    {
        self.call(move |state| Agent::Continue(func(state)))
    }
    /// The blocking variant of [`Handle::get_and_update`].
    ///
    /// ```
    /// use tokio_agent::{Agent, Result};
    ///
    /// fn main() -> Result<()> {
    ///     let agent = Agent::spawn_thread(|| 42).unwrap().blocking();
    ///     let result = agent.get_and_update(|x| {
    ///         *x += 1;
    ///         *x
    ///     })?;
    ///     assert_eq!(result, 43);
    ///
    ///     Ok(())
    /// }
    pub fn get_and_update<F, R>(&self, func: F) -> Result<R>
    where
        F: FnOnce(&mut T) -> R + Send + 'static,
        R: Send + 'static,
    {
        self.call(move |state| Agent::Continue(func(state)))
    }

    /// The blocking variant of [`Handle::update`].
    ///
    /// ```
    /// use tokio_agent::{Agent, Result};
    ///
    /// fn main() -> Result<()> {
    ///     let agent = Agent::spawn_thread(|| 42).unwrap().blocking();
    ///     agent.update(|x| { *x += 1; })?;
    ///     let result = agent.get(|x| *x)?;
    ///     assert_eq!(result, 43);
    ///     Ok(())
    /// }
    pub fn update<F>(&self, func: F) -> Result<()>
    where
        F: FnOnce(&mut T) + Send + 'static,
    {
        self.call(|x| Agent::Continue(func(x)))
    }

    /// The blocking variant of [`Handle::replace`].
    ///
    /// ```
    /// use tokio_agent::{Agent, Result};
    ///
    /// fn main() -> Result<()> {
    ///     let agent = Agent::spawn_thread(|| 42).unwrap().blocking();
    ///     let result = agent.replace(43)?;
    ///     assert_eq!(result, 42);
    ///     let result = agent.get(|x| *x)?;
    ///     assert_eq!(result, 43);
    ///     Ok(())
    /// }
    /// ```
    pub fn replace(&self, value: T) -> Result<T>
    where
        T: Send + 'static,
    {
        self.get_and_update(|state| std::mem::replace(state, value))
    }

    /// The blocking variant of [`Handle::take`].
    ///
    /// ```
    /// use tokio_agent::{Agent, Result};
    ///
    /// fn main() -> Result<()> {
    ///     let agent = Agent::spawn_thread(|| 42).unwrap().blocking();
    ///     let result = agent.take()?;
    ///     assert_eq!(result, 42);
    ///     let result = agent.get(|x| *x)?;
    ///     assert_eq!(result, 0);
    ///
    ///     Ok(())
    /// }
    /// ```
    pub fn take(&self) -> Result<T>
    where
        T: Default + Send + 'static,
    {
        self.get_and_update(|state| std::mem::take(state))
    }

    /// The blocking variant of [`Handle::stop`].
    ///
    /// ```
    /// use tokio_agent::{Agent, Result};
    ///
    /// fn main() -> Result<()> {
    ///     let agent = Agent::spawn_thread(|| 42).unwrap().blocking();
    ///     let agent_2 = agent.clone();
    ///
    ///     agent.stop()?;
    ///
    ///     assert!(agent_2.stop().is_err());
    ///
    ///     Ok(())
    /// }
    /// ```
    pub fn stop(self) -> Result<()> {
        self.call(|_| Agent::Stop(()))
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_agent_handle_drop_terminates_thread_agent() {
        let (sender, receiver) = oneshot::channel::<()>();

        let agent = Agent::spawn_thread(move || sender);
        drop(agent);

        // when the sender is dropped, this will return an Err()
        assert!(receiver.blocking_recv().is_err());
    }

    #[tokio::test]
    async fn test_agent_handle_drop_terminates_task_agent() {
        let (sender, receiver) = oneshot::channel::<()>();

        let agent = Agent::spawn_task(move || sender);
        drop(agent);

        // when the sender is dropped, this will return an Err()
        assert!(receiver.await.is_err());
    }
}
