import io
import os
import unittest
from contextlib import redirect_stdout

from raga_pipeline.audio import _resolve_demucs_device, _run_demucs_apply


class _FakeNoGrad:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakeTensor:
    def __init__(self) -> None:
        self.to_calls: list[str] = []

    def unsqueeze(self, _dim: int) -> "_FakeTensor":
        return self

    def to(self, device: str) -> "_FakeTensor":
        self.to_calls.append(device)
        return self


class _FakeModel:
    def __init__(self) -> None:
        self.to_calls: list[str] = []

    def to(self, device: str) -> "_FakeModel":
        self.to_calls.append(device)
        return self


class _FakeTorch:
    def __init__(self, *, cuda_available: bool, mps_available: bool, mps_built: bool) -> None:
        self._cuda_available = cuda_available
        self._mps_available = mps_available
        self._mps_built = mps_built

        class _Cuda:
            def __init__(self, outer: "_FakeTorch") -> None:
                self._outer = outer

            def is_available(self) -> bool:
                return self._outer._cuda_available

        class _Mps:
            def __init__(self, outer: "_FakeTorch") -> None:
                self._outer = outer

            def is_available(self) -> bool:
                return self._outer._mps_available

            def is_built(self) -> bool:
                return self._outer._mps_built

        class _Backends:
            def __init__(self, outer: "_FakeTorch") -> None:
                self.mps = _Mps(outer)

        self.cuda = _Cuda(self)
        self.backends = _Backends(self)

    def no_grad(self) -> _FakeNoGrad:
        return _FakeNoGrad()


class DemucsDeviceFallbackTests(unittest.TestCase):
    def test_auto_prefers_mps_when_cuda_unavailable(self) -> None:
        fake_torch = _FakeTorch(cuda_available=False, mps_available=True, mps_built=True)
        prev = os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK")
        try:
            os.environ.pop("PYTORCH_ENABLE_MPS_FALLBACK", None)
            selected = _resolve_demucs_device(None, fake_torch)
            self.assertEqual(selected, "mps")
            self.assertEqual(os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"), "1")
        finally:
            if prev is None:
                os.environ.pop("PYTORCH_ENABLE_MPS_FALLBACK", None)
            else:
                os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = prev

    def test_resolve_device_logs_fallback_capability(self) -> None:
        fake_torch = _FakeTorch(cuda_available=False, mps_available=True, mps_built=True)
        out = io.StringIO()
        with redirect_stdout(out):
            selected = _resolve_demucs_device("auto", fake_torch)
        text = out.getvalue()
        self.assertEqual(selected, "mps")
        self.assertIn("requested=auto, selected=mps", text)
        self.assertIn("PYTORCH_ENABLE_MPS_FALLBACK", text)

    def test_run_demucs_apply_retries_on_cpu_after_mps_failure(self) -> None:
        fake_torch = _FakeTorch(cuda_available=False, mps_available=True, mps_built=True)
        fake_model = _FakeModel()
        fake_wav = _FakeTensor()
        calls: list[str] = []

        def fake_apply_model(model, wav, device, progress):
            _ = (model, wav, progress)
            calls.append(device)
            if device == "mps":
                raise RuntimeError("mps kernel missing")
            return ["ok"]

        sources, effective_device = _run_demucs_apply(
            apply_model_fn=fake_apply_model,
            model=fake_model,
            wav=fake_wav,
            device="mps",
            torch_module=fake_torch,
        )

        self.assertEqual(sources, "ok")
        self.assertEqual(effective_device, "cpu")
        self.assertEqual(calls, ["mps", "cpu"])
        self.assertIn("cpu", fake_model.to_calls)


if __name__ == "__main__":
    unittest.main()
