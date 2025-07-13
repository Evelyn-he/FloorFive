I found it might be easier to just download the LLMs from the ONNX community on Hugging Face:  
https://huggingface.co/onnx-community

The link I used to download the Gemma model is:  
https://huggingface.co/onnx-community/gemma-3-1b-it-ONNX/blob/main/onnx/model_q4f16.onnx

They usually offer a list of quantized models as well. If the performance of Gemma is not as expected, we could also try models like Qwen 1.5B Instruct:  
https://huggingface.co/onnx-community/Qwen2.5-1.5B-Instruct/blob/main/onnx/model_q4f16.onnx

The script for using these models can be found in the Python notebook in `Gemma-3_1B.ipynb`. For reference you could check out https://github.com/DerrickJ1612/qnn_sample_apps. 
`qai_hub` is not required to run those. The notebook should be pretty general on the input models — we will just need to change the input model name and tweak a few lines to make things work.

However, when running on the provided machine, we will need to **uncomment** lines like:  
`# providers= [("QNNExecutionProvider", qnn_provider_options)],`  
to ensure it is using the QNN accelerator.
