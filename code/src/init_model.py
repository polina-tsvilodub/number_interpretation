from dotenv import load_dotenv
import openai
import os
from langchain_openai import ChatOpenAI
from langchain_openai.llms.base import OpenAI
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_anthropic import ChatAnthropic

from langchain_community.llms import OpenLLM
import random
from transformers import BitsAndBytesConfig
import anthropic
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI


def init_model(
        model_name, 
        load_as_quantized=False, 
        is_chat_model=False,
        local_url="http://localhost:2000/v1",
        **kwargs
    ):
    """
    Helper for initializing different models as the LLM backbone.
    May require API credentials.
    
    Parameters:
    ----------
    model_name: str
        Model to be used. Has to be a model known to langchain.
    load_as_quantized: bool
        Whether to load the model in quantized mode.
        Only applicable to HuggingFace model backbones.
    is_chat_model: bool
        Whether the model is a chat / instruction-tuned model.
        Only applicable to HuggingFace model backbones.
        Results in using message formatting that includes system
        messages and roles (system, user).
    local_url: str
        URL to the locally served model.
    **kwargs: dict
        LLM configs for sampling. Depend on particular model.

    Returns:
    --------
    model: langchain.LLM
        Initialized model.
    """
    load_dotenv()
    if "gpt" in model_name:
        try:
            
            openai_api_key = os.getenv("OPENAI_API_KEY")
        except:
            raise ValueError("OpenAI API key missing. Please add your API key to your env file with the key OPENAI_API_KEY")
        # load model
        model = ChatOpenAI(
            api_key=openai_api_key,
            model=model_name,
            **kwargs
        )
    elif "gemini" in model_name:
        
        os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
        # Create the model
        kwargs = kwargs if kwargs is not None else {}
        kwargs["response_mime_type"] = "text/plain"
        kwargs["timeout"] = None
        kwargs["max_retries"] = 2

        print("gemini kwargs ", kwargs)
        model = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
            **kwargs
        )
        
    elif "claude" in model_name:
        os.environ["ANTHROPIC_API_KEY"] = os.getenv("ANTHROPIC_API_KEY")
        model = ChatAnthropic(
            model=model_name,
            **kwargs
            # other params..
        )
    # huggingface models which usually have a / in their repo name
    elif "/" in model_name:
        load_dotenv()
        try:
            os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.environ["HUGGINGFACE_API_TOKEN"]
        except:
            try:
                assert os.environ["HUGGINGFACEHUB_API_TOKEN"] != "" 
            except:
                print((
                    "Huggingface API token is missing! "
                    "Add the token to your env file "
                    "with they key HUGGINGFACEHUB_API_TOKEN"
                ))
        if "llama" in model_name:
            print("\n ---- Trying to access the locally served model (running OpenLLM server required) ---- ")
            try:
                print("Running HF model via API")
                try:
                    if load_as_quantized:
                        quantization_config = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_quant_type="nf4",
                            bnb_4bit_compute_dtype="float16",
                            bnb_4bit_use_double_quant=True,
                        )
                        llm = HuggingFaceEndpoint(
                            repo_id=model_name,
                            task="text-generation",
                            quantization_config=quantization_config,
                            **kwargs
                        )
                    else:
                        llm = HuggingFaceEndpoint(
                            repo_id=model_name,
                            task="text-generation",
                            **kwargs
                        )
                    if is_chat_model:
                        model = ChatHuggingFace(llm=llm, verbose=True)
                    else:
                        model = llm
                except:
                    raise NotImplementedError(("Could not access the locally served model. " 
                                            "Please make sure the OpenLLM server is running the desired model. "
                                            "Please double check that the server url / port are correct."))
            except:
                model = OpenAI(
                    base_url=local_url,
                    api_key="na",
                    model=model_name,
                    # **kwargs
                )
        elif ("gemma" in model_name) or ("olmo" in model_name.lower()):
            # use huggingface pipeline 
            model = HuggingFaceEndpoint(
                repo_id=model_name,
                task="text-generation",
                **kwargs
            )
    else:
        raise ValueError((
            f"Unknown or incorrect model name {model_name}. "
             "See https://python.langchain.com/en/latest/modules"
             "/models/llms/integrations.html "
             "for a list of available models."
        ))
    return model
