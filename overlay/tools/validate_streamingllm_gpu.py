# SPDX-License-Identifier: Apache-2.0
# Added or modified for sparse_atten_spec validation; see NOTICE.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run baseline and self-spec in separate processes, then compare token IDs."""

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path


def graph_stats(worker):
    drafter = getattr(worker.model_runner, "drafter", None)
    wrappers = [
        getattr(drafter, name, None) for name in ("graph_runner", "random_graph_runner")
    ]
    return dict(
        captures=sum(
            entry.cudagraph is not None
            for wrapper in wrappers
            if wrapper is not None
            for entry in wrapper.concrete_cudagraph_entries.values()
        ),
        replays=getattr(drafter, "graph_replays", 0),
        padded_replays=getattr(drafter, "padded_graph_replays", 0),
        probabilistic_proposals=getattr(drafter, "probabilistic_proposals", 0),
    )


class VerificationWorkerExtension:
    def streamingllm_graph_stats(self):
        return graph_stats(self)

    def install_dynamic_checks(self):
        import torch

        from vllm.v1.sample.ops.topk_topp_sampler import (
            apply_top_k_top_p,
            apply_top_k_top_p_pytorch,
        )

        runner = self.model_runner
        self.dynamic_trace = []
        drafter = getattr(runner, "drafter", None)
        if drafter is None:
            return True
        topk_sampler = runner.sampler.topk_topp_sampler
        assert topk_sampler.forward.__func__ is topk_sampler.forward_native.__func__
        original_q = drafter.get_draft_probs

        def checked_q(meta, req_ids):
            counts = meta.num_draft_tokens
            cached = {rid: row for row, rid in enumerate(drafter.draft_req_ids)}
            for rid, count in zip(req_ids, counts):
                assert not count or rid in cached, (rid, count)
                if count:
                    row = cached[rid]
                    params = runner.requests[rid].sampling_params
                    saved = drafter.draft_sampling_metadata
                    if saved.temperature is not None:
                        assert (
                            abs(saved.temperature[row].item() - params.temperature)
                            < 1e-6
                        )
                    if saved.top_p is not None:
                        assert abs(saved.top_p[row].item() - params.top_p) < 1e-6
                    if saved.top_k is not None:
                        expected_k = (
                            params.top_k
                            if params.top_k > 0
                            else runner.model_config.get_vocab_size()
                        )
                        assert saved.top_k[row].item() == expected_k
            q = original_q(meta, req_ids)
            if q is not None:
                pieces = [
                    drafter.draft_hidden_states[cached[rid], :count]
                    for rid, count in zip(req_ids, counts)
                    if count
                ]
                # Reference transforms come from requests, not cached row metadata.
                logits = drafter.model.compute_logits(torch.cat(pieces)).float()
                offset = 0
                expected = []
                pytorch_expected = []
                raw_logits = logits.clone()
                for rid, count in zip(req_ids, counts):
                    if not count:
                        continue
                    params = runner.requests[rid].sampling_params
                    part = logits[offset : offset + count]
                    part = part / (params.temperature or 1.0)
                    p = torch.full((count,), params.top_p, device=part.device)
                    k = torch.full(
                        (count,),
                        params.top_k if params.top_k > 0 else part.shape[-1],
                        dtype=torch.int32,
                        device=part.device,
                    )
                    pytorch_expected.append(
                        apply_top_k_top_p_pytorch(part.clone(), k, p).softmax(-1)
                    )
                    expected.append(apply_top_k_top_p(part, k, p).softmax(-1))
                    offset += count
                reference = torch.cat(expected)
                pytorch_reference = torch.cat(pytorch_expected)
                if not torch.allclose(q, pytorch_reference, atol=2e-5, rtol=2e-4):
                    row = int((q - pytorch_reference).abs().max(-1).values.argmax())
                    changed = (q[row] > 0) != (pytorch_reference[row] > 0)
                    print(
                        "TOP_P_REFERENCE_DIFFERENCE",
                        dict(
                            row=row,
                            native_diff=(q - reference).abs().max().item(),
                            pytorch_diff=(q - pytorch_reference).abs().max().item(),
                            changed_tokens=changed.nonzero().flatten().tolist(),
                            changed_raw_logits=raw_logits[row, changed].tolist(),
                        ),
                        flush=True,
                    )
                assert q.shape == reference.shape
                if not torch.allclose(q, reference, atol=2e-5, rtol=2e-4):
                    from vllm.v1.spec_decode.sparse_attn.streamingllm import (
                        gather_draft_hidden_states,
                    )

                    mapping = torch.tensor(
                        [cached.get(rid, 0) for rid in req_ids],
                        device=q.device,
                        dtype=torch.int64,
                    )
                    gathered = gather_draft_hidden_states(
                        drafter.draft_hidden_states,
                        meta.cu_num_draft_tokens,
                        sum(counts),
                        len(req_ids),
                        mapping,
                    )
                    hidden_ref = torch.cat(pieces)
                    native = []
                    raw = drafter.model.compute_logits(hidden_ref).float()
                    offset = 0
                    for rid, count in zip(req_ids, counts):
                        if not count:
                            continue
                        params = runner.requests[rid].sampling_params
                        part = raw[offset : offset + count] / (
                            params.temperature or 1.0
                        )
                        native.append(
                            apply_top_k_top_p(
                                part,
                                torch.full(
                                    (count,),
                                    params.top_k,
                                    dtype=torch.int32,
                                    device=q.device,
                                ),
                                torch.full((count,), params.top_p, device=q.device),
                            ).softmax(-1)
                        )
                        offset += count
                    native = torch.cat(native)
                    index = int((q - reference).abs().max(-1).values.argmax())
                    print(
                        "Q_DIAGNOSTIC",
                        dict(
                            req_ids=req_ids,
                            counts=counts,
                            cached_ids=drafter.draft_req_ids,
                            cu_gpu=meta.cu_num_draft_tokens.tolist(),
                            hidden_equal=torch.equal(gathered, hidden_ref),
                            native_diff=(q - native).abs().max().item(),
                            pytorch_diff=(q - reference).abs().max().item(),
                            row=index,
                            q_top=q[index].topk(8).values.tolist(),
                            ref_top=reference[index].topk(8).values.tolist(),
                        ),
                        flush=True,
                    )
                assert torch.allclose(q, reference, atol=2e-5, rtol=2e-4), (
                    (q - reference).abs().max().item()
                )
                assert torch.isfinite(q).all() and torch.allclose(
                    q.sum(-1), torch.ones(q.shape[0], device=q.device), atol=1e-5
                )
            self.dynamic_trace.append(
                dict(
                    event="q",
                    req_ids=list(req_ids),
                    counts=list(counts),
                    cached_req_ids=list(drafter.draft_req_ids),
                    checked=q is not None,
                )
            )
            return q

        drafter.get_draft_probs = checked_q
        original_sample = runner._sample

        def checked_sample(logits, meta):
            result = original_sample(logits, meta)
            if meta is not None:
                ids = result.sampled_token_ids.cpu()
                lengths = (ids >= 0).sum(-1).tolist()
                counts = list(meta.num_draft_tokens)
                assert len(lengths) == len(counts)
                assert all(1 <= n <= count + 1 for n, count in zip(lengths, counts))
                self.dynamic_trace.append(
                    dict(
                        event="sample",
                        req_ids=runner.input_batch.req_ids.copy(),
                        counts=counts,
                        output_lengths=lengths,
                        tokens=ids.tolist(),
                    )
                )
            return result

        runner._sample = checked_sample
        return True

    def collect_dynamic_checks(self):
        return self.dynamic_trace


def run_dynamic(
    model,
    mode,
    output,
    mixed,
    unseeded=False,
    sparse_min_tokens=32,
    algorithm="streamingllm",
):
    assert sparse_min_tokens >= 32
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"
    from vllm import LLM, SamplingParams

    kwargs = dict(
        model=model,
        dtype="bfloat16",
        enforce_eager=mode != "graph",
        attention_config={"backend": "FLASH_ATTN"},
        max_model_len=2048,
        max_num_seqs=4,
        max_num_batched_tokens=256,
        enable_chunked_prefill=True,
        enable_prefix_caching=False,
        async_scheduling=False,
        kv_cache_memory_bytes=1024**3,
        seed=42,
        disable_log_stats=False,
        worker_extension_cls="validate_streamingllm_gpu.VerificationWorkerExtension",
    )
    if mode == "graph":
        kwargs["compilation_config"] = dict(
            mode=3,
            cudagraph_mode="FULL_DECODE_ONLY",
            cudagraph_capture_sizes=[1, 2, 4, 5, 10, 20],
        )
    if mode != "baseline":
        kwargs["speculative_config"] = dict(
            method="sparse_attn",
            sparse_attn_algorithm=algorithm,
            num_speculative_tokens=4,
            sparse_attn_ratio=0.07,
            sparse_attn_min_tokens=sparse_min_tokens,
        )
    llm = LLM(**kwargs)
    llm.collective_rpc("install_dynamic_checks")
    tokens = llm.get_tokenizer().encode(
        "The city has a library and a garden. Explain their history. "
    )
    lengths = [15, 511, 63, 1023, 300, 2040]
    limits = [7, 45, 29, 97, 63, 8]
    arrivals = [0, 0, 3, 7, 11, 19]
    names = {}
    outputs = {}
    history = []
    for step in range(512):
        for i, arrival in enumerate(arrivals):
            if step != arrival:
                continue
            random = mixed and i % 2 == 1
            params = SamplingParams(
                temperature=[0.6, 0.8, 1.0][i // 2] if random else 0,
                top_p=[0.95, 0.8, 0.9][i // 2],
                top_k=[20, 10, 50][i // 2],
                max_tokens=limits[i],
                ignore_eos=True,
                seed=None if unseeded else 100 + i,
            )
            prompt = {
                "prompt_token_ids": (tokens * (1 + lengths[i] // len(tokens)))[
                    : lengths[i]
                ]
            }
            rid = llm.llm_engine.add_request(f"dynamic-{i}", prompt, params)
            names[rid] = str(i)
            names[f"dynamic-{i}"] = str(i)
            history.append(dict(step=step, event="add", request=str(i), random=random))
        if llm.llm_engine.has_unfinished_requests():
            for result in llm.llm_engine.step():
                if result.finished:
                    name = names[result.request_id]
                    assert name not in outputs
                    outputs[name] = list(result.outputs[0].token_ids)
                    history.append(dict(step=step, event="finish", request=name))
        if step >= max(arrivals) and not llm.llm_engine.has_unfinished_requests():
            break
    else:
        raise AssertionError("Dynamic requests did not finish within 512 steps")
    assert len(outputs) == 6 and all(
        len(outputs[str(i)]) == limits[i] for i in range(6)
    )
    trace = llm.collective_rpc("collect_dynamic_checks")
    graphs = llm.collective_rpc("streamingllm_graph_stats")
    if mode != "baseline":
        samples = [row for row in trace[0] if row["event"] == "sample"]
        assert samples and any(c > 0 for row in samples for c in row["counts"])
        if mixed:
            assert any(row["checked"] for row in trace[0] if row["event"] == "q")
        if mode == "graph":
            assert graphs[0]["replays"] > 0 and graphs[0]["padded_replays"] > 0
    Path(output).write_text(
        json.dumps(
            dict(
                mode=mode,
                algorithm=algorithm,
                mixed=mixed,
                unseeded=unseeded,
                engine_config=kwargs,
                tokens=outputs,
                history=history,
                trace=trace,
                graphs=graphs,
            ),
            indent=2,
        )
    )
    print(
        "PASS: dynamic requests completed;",
        mode,
        "mixed=",
        mixed,
        "graphs=",
        graphs,
        flush=True,
    )


def run(model, mode, output, algorithm="streamingllm"):
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"
    from vllm import LLM, SamplingParams

    kwargs = dict(
        model=model,
        dtype="bfloat16",
        enforce_eager=mode != "graph",
        attention_config={"backend": "FLASH_ATTN"},
        max_model_len=2048,
        max_num_seqs=4,
        max_num_batched_tokens=256,
        enable_chunked_prefill=True,
        enable_prefix_caching=False,
        async_scheduling=False,
        gpu_memory_utilization=0.7,
        disable_log_stats=False,
        seed=0,
        worker_extension_cls=("validate_streamingllm_gpu.VerificationWorkerExtension"),
    )
    if mode == "graph":
        # Dynamic compilation covers draft sizes excluded by target padding.
        kwargs["compilation_config"] = dict(
            mode=3,
            cudagraph_mode="FULL_DECODE_ONLY",
            cudagraph_capture_sizes=[1, 4, 8, 16],
        )
    if mode in ("spec", "graph"):
        kwargs["speculative_config"] = dict(
            method="sparse_attn",
            sparse_attn_algorithm=algorithm,
            num_speculative_tokens=3,
            sparse_attn_ratio=0.05,
            sparse_attn_min_tokens=32,
        )
    llm = LLM(**kwargs)
    seed_tokens = llm.get_tokenizer().encode(
        "The city has a library, a garden and a university. Explain their history. "
    )
    lengths = (15, 63, 511, 1023, 2040, 300)
    prompts = [
        {"prompt_token_ids": (seed_tokens * (1 + n // len(seed_tokens)))[:n]}
        for n in lengths
    ]
    params = [
        SamplingParams(temperature=0, max_tokens=min(limit, 2048 - n), ignore_eos=True)
        for n, limit in zip(lengths, (7, 29, 128, 97, 8, 45))
    ]
    outputs = llm.generate(prompts, params, use_tqdm=False)
    metrics = {
        name: sum(metric.value for metric in llm.get_metrics() if metric.name == name)
        for name in (
            "vllm:spec_decode_num_draft_tokens",
            "vllm:spec_decode_num_accepted_tokens",
        )
    }
    Path(output).write_text(
        json.dumps(
            dict(
                algorithm=algorithm,
                tokens=[list(result.outputs[0].token_ids) for result in outputs],
                metrics=metrics,
                graphs=llm.collective_rpc("streamingllm_graph_stats"),
            )
        )
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--algorithm", choices=("streamingllm", "vegas"), default="streamingllm"
    )
    parser.add_argument("--mode", choices=("baseline", "spec", "graph"))
    parser.add_argument("--output")
    parser.add_argument("--dynamic", action="store_true")
    parser.add_argument("--mixed", action="store_true")
    parser.add_argument("--unseeded", action="store_true")
    parser.add_argument("--sparse-min-tokens", type=int, default=32)
    args = parser.parse_args()
    if args.dynamic:
        assert args.mode and args.output
        run_dynamic(
            args.model,
            args.mode,
            args.output,
            args.mixed,
            args.unseeded,
            args.sparse_min_tokens,
            args.algorithm,
        )
        return
    if args.mode:
        assert args.output
        run(args.model, args.mode, args.output, args.algorithm)
        return
    with tempfile.TemporaryDirectory(prefix="streamingllm-") as directory:
        results = []
        for mode in ("baseline", "spec", "graph"):
            path = Path(directory) / f"{mode}.json"
            subprocess.run(
                [
                    os.sys.executable,
                    __file__,
                    "--model",
                    args.model,
                    "--algorithm",
                    args.algorithm,
                    "--mode",
                    mode,
                    "--output",
                    str(path),
                ],
                check=True,
            )
            results.append(json.loads(path.read_text()))
        assert results[0]["tokens"] == results[1]["tokens"] == results[2]["tokens"], (
            "Baseline/eager self-spec/CUDA graph token IDs differ"
        )
        for mode, result in zip(("spec", "graph"), results[1:]):
            metrics = result["metrics"]
            drafted = metrics["vllm:spec_decode_num_draft_tokens"]
            accepted = metrics["vllm:spec_decode_num_accepted_tokens"]
            assert drafted > 0, f"{mode}: no draft tokens were verified"
            assert drafted > accepted, f"{mode}: no rejected suffix was exercised"
            print(f"{mode}: drafted={drafted}, accepted={accepted}")
        graphs = results[2]["graphs"]
        assert sum(g["captures"] for g in graphs) > 0, "No draft graph was captured"
        assert sum(g["replays"] for g in graphs) > 1, "Draft graph did not replay"
        assert sum(g["padded_replays"] for g in graphs) > 0, (
            "No padded replay exercised"
        )
        print(f"PASS: identical token IDs across all 3 modes; draft graphs={graphs}")


if __name__ == "__main__":
    main()
