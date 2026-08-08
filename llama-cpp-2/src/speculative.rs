//! Experimental wrappers for llama.cpp speculative decoding helpers.
//!
//! MTP decoding follows this sequence: call [`MtpSpeculative::reset`] for a
//! new request, process target prefill batches, call [`MtpSpeculative::begin`],
//! create a bounded draft, decode its tokens in the target context, process
//! that target batch, then accept the number of tokens the target retained. A
//! non-empty draft must be accepted before the next draft operation.

use std::ptr::NonNull;

use crate::context::LlamaContext;
use crate::llama_batch::LlamaBatch;
use crate::status_is_ok;
use crate::token::LlamaToken;

/// Parameters for same-model MTP speculative decoding.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MtpSpeculativeParams {
    /// Maximum number of draft tokens to propose.
    pub n_max: i32,
    /// Minimum number of draft tokens required before returning a draft.
    pub n_min: i32,
    /// Minimum draft probability accepted by llama.cpp's MTP drafter.
    pub p_min: f32,
}

impl Default for MtpSpeculativeParams {
    fn default() -> Self {
        Self {
            n_max: 3,
            n_min: 0,
            p_min: 0.0,
        }
    }
}

/// Errors returned by the MTP speculative wrapper.
#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum MtpSpeculativeError {
    /// Invalid parameters were provided.
    #[error("invalid MTP speculative parameters")]
    InvalidParams,
    /// llama.cpp returned a null speculative handle.
    #[error("llama.cpp failed to initialize MTP speculative decoding")]
    InitFailed,
    /// llama.cpp rejected a wrapper call.
    #[error("llama.cpp MTP speculative call failed with status {0}")]
    Status(i32),
    /// The operation is not valid in the current speculative-decoding state.
    #[error("invalid MTP speculative operation for the current state")]
    InvalidState,
    /// The draft output exceeded the caller-provided bound.
    #[error("llama.cpp MTP draft exceeded configured maximum")]
    DraftOverflow,
}

/// RAII owner for a same-model MTP speculative context.
///
/// This wrapper currently binds llama.cpp's speculative state to sequence 0.
/// Batches passed to [`Self::process`] must therefore contain only sequence 0.
#[derive(Debug)]
pub struct MtpSpeculative<'model> {
    raw: NonNull<llama_cpp_sys_2::llama_rs_mtp_speculative>,
    target_context: LlamaContext<'model>,
    draft_context: LlamaContext<'model>,
    state: MtpSpeculativeState,
}

impl<'model> MtpSpeculative<'model> {
    /// Create a new MTP speculative helper from a target context and an MTP
    /// draft context.
    ///
    /// # Errors
    ///
    /// Returns an error if parameters are invalid or llama.cpp cannot
    /// initialize the speculative implementation for the loaded model.
    pub fn new(
        target_context: LlamaContext<'model>,
        draft_context: LlamaContext<'model>,
        params: MtpSpeculativeParams,
    ) -> Result<Self, MtpSpeculativeError> {
        let n_max = validate_params(params)?;

        let raw = unsafe {
            llama_cpp_sys_2::llama_rs_mtp_speculative_init(
                target_context.context.as_ptr(),
                draft_context.context.as_ptr(),
                params.n_max,
                params.n_min,
                params.p_min,
            )
        };
        let raw = NonNull::new(raw).ok_or(MtpSpeculativeError::InitFailed)?;

        Ok(Self {
            raw,
            target_context,
            draft_context,
            state: MtpSpeculativeState::new(n_max),
        })
    }

    /// Access the target context.
    #[must_use]
    pub fn target_context(&self) -> &LlamaContext<'model> {
        &self.target_context
    }

    /// Access the target context for decode and cache rollback operations.
    pub fn target_context_mut(&mut self) -> &mut LlamaContext<'model> {
        &mut self.target_context
    }

    /// Access the draft context for cache rollback operations.
    pub fn draft_context_mut(&mut self) -> &mut LlamaContext<'model> {
        &mut self.draft_context
    }

    /// Reset speculative state before beginning a new request.
    ///
    /// This preserves the target and draft contexts, but recreates the
    /// llama.cpp speculative helper so hidden-state carryover cannot cross
    /// request boundaries. Call this before processing target prefill batches.
    ///
    /// # Errors
    ///
    /// Returns an error if llama.cpp cannot recreate the speculative helper.
    pub fn reset(&mut self) -> Result<(), MtpSpeculativeError> {
        let status = unsafe { llama_cpp_sys_2::llama_rs_mtp_speculative_reset(self.raw.as_ptr()) };
        status_to_result(status)?;
        self.state.reset();
        Ok(())
    }

    /// Begin drafting after target prefill has been processed.
    ///
    /// Call [`Self::reset`] before processing a new request's prefill batches.
    ///
    /// # Errors
    ///
    /// Returns an error if llama.cpp rejects the call.
    pub fn begin(&mut self, prompt_tokens: &[LlamaToken]) -> Result<(), MtpSpeculativeError> {
        let prompt = tokens_to_raw(prompt_tokens);
        let status = unsafe {
            llama_cpp_sys_2::llama_rs_mtp_speculative_begin(
                self.raw.as_ptr(),
                prompt.as_ptr(),
                prompt.len(),
            )
        };
        status_to_result(status)?;
        self.state.begin();
        Ok(())
    }

    /// Process a batch that was just decoded by the target context.
    ///
    /// The batch must contain token input for sequence 0 only.
    ///
    /// # Errors
    ///
    /// Returns an error if llama.cpp cannot update the MTP draft context.
    pub fn process(&mut self, batch: &LlamaBatch<'_>) -> Result<(), MtpSpeculativeError> {
        self.state.process()?;
        let status = unsafe {
            llama_cpp_sys_2::llama_rs_mtp_speculative_process(
                self.raw.as_ptr(),
                std::ptr::from_ref(&batch.llama_batch),
            )
        };
        status_to_result(status)
    }

    /// Generate up to `max_draft_tokens` draft tokens after `id_last`.
    ///
    /// # Errors
    ///
    /// Returns an error if llama.cpp rejects the draft operation or emits more
    /// draft tokens than requested. `max_draft_tokens` must be between one
    /// and the `n_max` specified at construction.
    pub fn draft(
        &mut self,
        n_past: i32,
        id_last: LlamaToken,
        prompt_tokens: &[LlamaToken],
        max_draft_tokens: u16,
    ) -> Result<Vec<LlamaToken>, MtpSpeculativeError> {
        if n_past < 0 {
            return Err(MtpSpeculativeError::InvalidParams);
        }
        self.state.draft(max_draft_tokens)?;

        let prompt = tokens_to_raw(prompt_tokens);
        let mut raw_out = vec![0; usize::from(max_draft_tokens)];
        let mut out_len = 0_usize;
        let status = unsafe {
            llama_cpp_sys_2::llama_rs_mtp_speculative_draft(
                self.raw.as_ptr(),
                n_past,
                id_last.0,
                prompt.as_ptr(),
                prompt.len(),
                max_draft_tokens,
                raw_out.as_mut_ptr(),
                raw_out.len(),
                &raw mut out_len,
            )
        };
        if status == llama_cpp_sys_2::LLAMA_RS_STATUS_ALLOCATION_FAILED {
            return Err(MtpSpeculativeError::DraftOverflow);
        }
        status_to_result(status)?;
        if out_len > raw_out.len() {
            return Err(MtpSpeculativeError::DraftOverflow);
        }
        let draft_len = u16::try_from(out_len).map_err(|_| MtpSpeculativeError::DraftOverflow)?;
        raw_out.truncate(out_len);
        self.state.draft_completed(draft_len);
        Ok(raw_out.into_iter().map(LlamaToken).collect())
    }

    /// Notify llama.cpp how many draft tokens the target context accepted.
    ///
    /// # Errors
    ///
    /// Returns an error if llama.cpp rejects the call.
    pub fn accept(&mut self, n_accepted: u16) -> Result<(), MtpSpeculativeError> {
        self.state.accept(n_accepted)?;
        let status = unsafe {
            llama_cpp_sys_2::llama_rs_mtp_speculative_accept(self.raw.as_ptr(), n_accepted)
        };
        status_to_result(status)?;
        self.state.accept_completed();
        Ok(())
    }
}

impl Drop for MtpSpeculative<'_> {
    fn drop(&mut self) {
        unsafe {
            llama_cpp_sys_2::llama_rs_mtp_speculative_free(self.raw.as_ptr());
        }
    }
}

fn tokens_to_raw(tokens: &[LlamaToken]) -> Vec<llama_cpp_sys_2::llama_token> {
    tokens.iter().map(|token| token.0).collect()
}

fn validate_params(params: MtpSpeculativeParams) -> Result<u16, MtpSpeculativeError> {
    if params.n_max <= 0
        || params.n_max > i32::from(u16::MAX)
        || params.n_min < 0
        || params.n_min > params.n_max
        || !params.p_min.is_finite()
        || !(0.0..=1.0).contains(&params.p_min)
    {
        return Err(MtpSpeculativeError::InvalidParams);
    }

    u16::try_from(params.n_max).map_err(|_| MtpSpeculativeError::InvalidParams)
}

#[derive(Debug, Eq, PartialEq)]
enum MtpSpeculativeState {
    New { n_max: u16 },
    Ready { n_max: u16 },
    DraftPending { n_max: u16, draft_len: u16 },
}

impl MtpSpeculativeState {
    const fn new(n_max: u16) -> Self {
        Self::New { n_max }
    }

    fn begin(&mut self) {
        *self = Self::Ready {
            n_max: self.n_max(),
        };
    }

    fn reset(&mut self) {
        *self = Self::New {
            n_max: self.n_max(),
        };
    }

    fn process(&self) -> Result<(), MtpSpeculativeError> {
        if matches!(
            self,
            Self::New { .. } | Self::Ready { .. } | Self::DraftPending { .. }
        ) {
            Ok(())
        } else {
            Err(MtpSpeculativeError::InvalidState)
        }
    }

    fn draft(&self, max_draft_tokens: u16) -> Result<(), MtpSpeculativeError> {
        let Self::Ready { n_max } = self else {
            return Err(MtpSpeculativeError::InvalidState);
        };
        if max_draft_tokens == 0 || max_draft_tokens > *n_max {
            return Err(MtpSpeculativeError::InvalidParams);
        }
        Ok(())
    }

    fn draft_completed(&mut self, draft_len: u16) {
        let n_max = self.n_max();
        debug_assert!(draft_len <= n_max);
        *self = if draft_len == 0 {
            Self::Ready { n_max }
        } else {
            Self::DraftPending { n_max, draft_len }
        };
    }

    fn accept(&self, n_accepted: u16) -> Result<(), MtpSpeculativeError> {
        let Self::DraftPending { draft_len, .. } = self else {
            return Err(MtpSpeculativeError::InvalidState);
        };
        if n_accepted > *draft_len {
            return Err(MtpSpeculativeError::InvalidParams);
        }
        Ok(())
    }

    fn accept_completed(&mut self) {
        *self = Self::Ready {
            n_max: self.n_max(),
        };
    }

    const fn n_max(&self) -> u16 {
        match self {
            Self::New { n_max } | Self::Ready { n_max } | Self::DraftPending { n_max, .. } => {
                *n_max
            }
        }
    }
}

fn status_to_result(status: llama_cpp_sys_2::llama_rs_status) -> Result<(), MtpSpeculativeError> {
    if status_is_ok(status) {
        Ok(())
    } else {
        Err(MtpSpeculativeError::Status(status as i32))
    }
}

#[cfg(test)]
mod tests {
    use super::{MtpSpeculativeError, MtpSpeculativeParams, MtpSpeculativeState};

    #[test]
    fn params_reject_drafts_that_cannot_be_accepted_as_u16() {
        let params = MtpSpeculativeParams {
            n_max: i32::from(u16::MAX) + 1,
            ..MtpSpeculativeParams::default()
        };

        assert_eq!(
            super::validate_params(params),
            Err(MtpSpeculativeError::InvalidParams)
        );
    }

    #[test]
    fn state_allows_prefill_processing_but_requires_begin_before_draft() {
        let state = MtpSpeculativeState::new(3);

        assert_eq!(state.process(), Ok(()));
        assert_eq!(state.draft(1), Err(MtpSpeculativeError::InvalidState));
    }

    #[test]
    fn state_enforces_the_per_call_draft_bound() {
        let mut state = MtpSpeculativeState::new(3);
        state.begin();

        assert_eq!(state.draft(0), Err(MtpSpeculativeError::InvalidParams));
        assert_eq!(state.draft(4), Err(MtpSpeculativeError::InvalidParams));
        assert_eq!(state.draft(3), Ok(()));
    }

    #[test]
    fn state_rejects_accept_outside_the_pending_draft_boundary() {
        let mut state = MtpSpeculativeState::new(u16::MAX);
        state.begin();
        state.draft(u16::MAX).unwrap();

        assert_eq!(
            state.accept(u16::MAX),
            Err(MtpSpeculativeError::InvalidState)
        );
        state.draft_completed(u16::MAX);
        assert_eq!(state.accept(u16::MAX), Ok(()));
        state.accept_completed();
        assert_eq!(state.accept(0), Err(MtpSpeculativeError::InvalidState));
    }

    #[test]
    fn state_rejects_accepting_more_tokens_than_were_drafted() {
        let mut state = MtpSpeculativeState::new(3);
        state.begin();
        state.draft(2).unwrap();
        state.draft_completed(2);

        assert_eq!(state.accept(3), Err(MtpSpeculativeError::InvalidParams));
        assert_eq!(state.accept(0), Ok(()));
    }

    #[test]
    fn state_keeps_a_draft_pending_while_processing_the_target_batch() {
        let mut state = MtpSpeculativeState::new(3);
        state.begin();
        state.draft(2).unwrap();
        state.draft_completed(2);

        assert_eq!(state.process(), Ok(()));
        assert_eq!(state.accept(2), Ok(()));
    }

    #[test]
    fn reset_discards_a_previous_request_pending_draft() {
        let mut state = MtpSpeculativeState::new(3);
        state.begin();
        state.draft(2).unwrap();
        state.draft_completed(2);

        state.reset();

        assert_eq!(state.accept(2), Err(MtpSpeculativeError::InvalidState));
        assert_eq!(state.draft(3), Err(MtpSpeculativeError::InvalidState));
        assert_eq!(state.process(), Ok(()));
        state.begin();
        assert_eq!(state.draft(3), Ok(()));
    }
}
