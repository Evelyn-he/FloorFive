import numpy as np
import onnxruntime as ort

def create_ort_session(model_path: str):
    # cache our model to reduce load times
    options = ort.SessionOptions()
    #options.add_session_config_entry("ep.context_enable", 1)

    session = ort.InferenceSession(
        model_path,
        sess_options = options,
        providers = ["QNNExecutionProvider"],
        provider_options = [
            {
                "backend_path": "QnnHtp.dll",
                "htp_performance_mode": "burst",
                "htp_graph_finalization_optimization_mode": "3",
            }
        ]
    )

    return session

# Dummy example
# image = np.random.randn(1,3,192,192).astype(np.uint8)
# sess = create_ort_session('mediapipe_face-facelandmarkdetector-w8a8.onnx')
# out = sess.run(None, {"image": image})
# print(out)
# print("done")