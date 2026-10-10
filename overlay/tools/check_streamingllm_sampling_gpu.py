# SPDX-License-Identifier: Apache-2.0
# Added or modified for sparse_atten_spec validation; see NOTICE.
"""Statistical regression for native sampling and hidden-state q reconstruction."""

from types import SimpleNamespace

import torch

from vllm.v1.sample.logits_processor import LogitsProcessors
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p_pytorch
from vllm.v1.sample.rejection_sampler import rejection_sample
from vllm.v1.sample.sampler import Sampler
from vllm.v1.spec_decode.sparse_attn.proposer import SparseAttnProposer
from vllm.v1.worker.gpu_model_runner import GPUModelRunner


def make_drafter(batch):
    empty = torch.empty(0, device="cuda")
    metadata = SamplingMetadata(
        temperature=torch.full((batch,), 0.6, device="cuda"),
        all_greedy=False,
        all_random=True,
        top_p=torch.full((batch,), 0.95, device="cuda"),
        top_k=torch.full((batch,), 2, dtype=torch.int32, device="cuda"),
        generators={},
        max_num_logprobs=None,
        no_penalties=True,
        prompt_token_ids=None,
        frequency_penalties=empty,
        presence_penalties=empty,
        repetition_penalties=empty,
        output_token_ids=[],
        allowed_token_ids_mask=None,
        bad_words_token_ids={},
        logitsprocs=LogitsProcessors(),
    )
    drafter = SparseAttnProposer.__new__(SparseAttnProposer)
    drafter.runner = SimpleNamespace(
        input_batch=SimpleNamespace(sampling_metadata=metadata), sampler=Sampler()
    )
    # A real LM head allocates logits; sampling must not alias the hidden cache.
    drafter.model = SimpleNamespace(compute_logits=lambda hidden: hidden.clone())
    drafter.num_speculative_tokens = 1
    drafter.draft_req_ids = [str(i) for i in range(batch)]
    drafter.draft_sampling_metadata = metadata
    drafter.query_start_loc = torch.arange(batch + 1, dtype=torch.int32, device="cuda")
    return drafter, metadata


def main():
    torch.manual_seed(42)
    batch = 40000
    drafter, metadata = make_drafter(batch)
    logits = torch.tensor([0.70, 0.18, 0.08, 0.04], device="cuda").log()
    drafter.draft_hidden_states = logits.expand(batch, 1, -1).clone()
    original_hidden = drafter.draft_hidden_states.clone()
    draft_ids = drafter.runner.sampler(
        logits.expand(batch, -1).clone(), metadata
    ).sampled_token_ids.flatten()
    probs = drafter.get_draft_probs(
        SimpleNamespace(
            num_draft_tokens=[1] * batch,
            cu_num_draft_tokens=drafter.query_start_loc[1:],
        ),
        drafter.draft_req_ids,
    )
    expected = apply_top_k_top_p_pytorch(
        logits.expand(batch, -1).clone() / 0.6, metadata.top_k, metadata.top_p
    ).softmax(-1)
    assert torch.allclose(probs, expected, atol=1e-6)
    assert torch.equal(drafter.draft_hidden_states, original_hidden)
    original = probs.clone()
    target = torch.tensor([0.1, 0.2, 0.3, 0.4], device="cuda")
    outputs = rejection_sample(
        draft_ids.to(torch.int32).contiguous(),
        [1] * batch,
        1,
        drafter.query_start_loc[1:],
        probs.contiguous(),
        target.log().expand(batch, -1).contiguous(),
        torch.zeros((batch, 1), dtype=torch.int32, device="cuda"),
        metadata,
    )
    observed = torch.bincount(outputs[:, 0].long(), minlength=4).float() / batch
    assert (observed - target).abs().max() < 0.012, observed
    assert torch.equal(probs, original), "Proposal probabilities mutated"
    print("PASS: 40,000 p/q trials; observed=", observed.tolist(), flush=True)
    # Native mixed batches: greedy uses argmax; only random rows use p/q.
    metadata.all_random = False
    metadata.temperature[::2] = 0
    draft_ids = drafter.runner.sampler(
        logits.expand(batch, -1).clone(), metadata
    ).sampled_token_ids.flatten()
    assert torch.all(draft_ids[::2] == logits.argmax())
    probs = drafter.get_draft_probs(
        SimpleNamespace(
            num_draft_tokens=[1] * batch,
            cu_num_draft_tokens=drafter.query_start_loc[1:],
        ),
        drafter.draft_req_ids,
    )
    assert torch.allclose(probs[1::2], expected[1::2], atol=1e-6)
    assert torch.equal(drafter.draft_hidden_states, original_hidden)
    outputs = rejection_sample(
        draft_ids.to(torch.int32).contiguous(),
        [1] * batch,
        1,
        drafter.query_start_loc[1:],
        probs.contiguous(),
        target.log().expand(batch, -1).contiguous(),
        torch.zeros((batch, 1), dtype=torch.int32, device="cuda"),
        metadata,
    )
    assert torch.all(outputs[::2, 0] == target.argmax())
    random_observed = torch.bincount(outputs[1::2, 0].long(), minlength=4).float() / (
        batch // 2
    )
    assert (random_observed - target).abs().max() < 0.015, random_observed
    print(
        "PASS: mixed greedy/random native rejection; random=",
        random_observed.tolist(),
        flush=True,
    )
    # Exercise the production request-ID remapping and truncated draft counts.
    cached = torch.arange(24, device="cuda").view(2, 3, 4).float()
    runner = SimpleNamespace(
        _draft_probs=cached,
        _draft_prob_req_ids=["a", "b"],
        input_batch=SimpleNamespace(req_ids=["b", "a"]),
    )
    remapped = GPUModelRunner._get_spec_decode_draft_probs(
        runner, SimpleNamespace(num_draft_tokens=[2, 1])
    )
    assert torch.equal(remapped, torch.cat([cached[1, :2], cached[0, :1]]))
    drafter, metadata = make_drafter(2)
    drafter.num_speculative_tokens = 3
    drafter.draft_hidden_states = cached
    spec = SimpleNamespace(
        num_draft_tokens=[2, 1],
        cu_num_draft_tokens=torch.tensor([2, 3], dtype=torch.int32, device="cuda"),
    )
    actual = drafter.get_draft_probs(spec, ["1", "0"])
    expected_logits = torch.cat([cached[1, :2], cached[0, :1]]) / 0.6
    expected = apply_top_k_top_p_pytorch(
        expected_logits, metadata.top_k[[1, 1, 0]], metadata.top_p[[1, 1, 0]]
    ).softmax(-1)
    assert torch.allclose(actual, expected, atol=1e-6)
    spec.num_draft_tokens = [0, 2]
    spec.cu_num_draft_tokens.copy_(
        torch.tensor([0, 2], dtype=torch.int32, device="cuda")
    )
    assert torch.allclose(
        drafter.get_draft_probs(spec, ["new", "1"]), expected[:2], atol=1e-6
    )
    print(
        "PASS: production hidden gather/q reconstruction with reorder and new zero row",
        flush=True,
    )
    drafter, metadata = make_drafter(256)
    first = drafter.runner.sampler(
        logits.expand(256, -1).clone(), metadata
    ).sampled_token_ids
    second = drafter.runner.sampler(
        logits.expand(256, -1).clone(), metadata
    ).sampled_token_ids
    assert not torch.equal(first, second), "Native sampler froze the random stream"
    print("PASS: native sampler RNG advances outside forward graphs", flush=True)
    # BF16-sized ties at a top-p cutoff: q must match the actual draft sampler.
    drafter, metadata = make_drafter(20000)
    metadata.temperature.fill_(0.8)
    metadata.top_p.fill_(0.8)
    metadata.top_k.fill_(3)
    tied_logits = torch.tensor([20.5, 19.375, 19.375, -20.0], device="cuda")
    drafter.draft_hidden_states = tied_logits.expand(20000, 1, -1).clone()
    q = drafter.get_draft_probs(
        SimpleNamespace(
            num_draft_tokens=[1] * 20000,
            cu_num_draft_tokens=drafter.query_start_loc[1:],
        ),
        drafter.draft_req_ids,
    )
    sampled = drafter.runner.sampler(
        tied_logits.expand(20000, -1).clone(), metadata
    ).sampled_token_ids.flatten()
    observed = torch.bincount(sampled.long(), minlength=4).float() / 20000
    expected = q.mean(0)
    print(
        "TOP_P_TIE_CHECK",
        dict(q=expected.tolist(), observed=observed.tolist()),
        flush=True,
    )
    assert (observed - expected).abs().max() < 0.015, (
        "Actual draft sampler distribution differs from reconstructed q at ties"
    )
    print("PASS: unseeded draft sampler agrees with q at top-p ties", flush=True)


if __name__ == "__main__":
    main()
