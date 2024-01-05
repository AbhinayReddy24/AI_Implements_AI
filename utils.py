from asyncore import read
import os
from dotenv import load_dotenv, find_dotenv

import numpy as np
import nest_asyncio
from trulens_eval import Feedback

nest_asyncio.apply()


def get_openai_api_key():
    _ = load_dotenv(find_dotenv())

    return os.getenv("OPENAI_API_KEY")

from trulens_eval.feedback import Groundedness

from trulens_eval import (
    OpenAI,
    Feedback,
    TruLlama
)

def get_prebuilt_trulens_recorder(query_engine, app_id):
    openai = OpenAI()

    qa_relevance = (
        Feedback(
            openai.relevance_with_cot_reasons, name = "answer relevance"
        ).on_input_output()
        )
    
    qs_relevance = (
        Feedback(
            openai.relevance_with_cot_reasons, name = "context relecnace"
        ).on_input()
        .on(TruLlama.select_source_nodes().node.text)
        .aggregate(np.mean)
    )

    grounded = Groundedness(groundedness_provider = openai)

    groundedness = (
        Feedback(
            grounded.groundedness_measure_with_cot_reasons, name = "groundedness"
        ).on_output()
        .aggregate(grounded.grounded_statements_aggregator)
    )

    feedbacks = [qa_relevance, qs_relevance, groundedness]
    tru_recorder = TruLlama(
        query_engine, 
        app_id =  app_id,
        feedbacks = feedbacks
    )

    return tru_recorder


from llama_index import ServiceContext, VectorStoreIndex, StorageContext
from llama_index.node_parser import SentenceWindowNodeParser
from llama_index.indices.postprocessor import MetadataReplacementPostProcessor
from llama_index.indices.postprocessor import SentenceTransformerRerank
from llama_index import load_index_from_storage

import os

def build_sentence_window_index(
    documents, 
    llm,
    sentence_window_size = 3, 
    embed_model="local:BAAI/bge-small-en-v1.5", 
    save_dir="sentence_index"
):
    # create the sentence window node parser w/ default settings
    node_parser = SentenceWindowNodeParser.from_defaults(
        window_size=sentence_window_size,
        window_metadata_key="window",
        original_text_metadata_key="original_text",
    )
    sentence_context = ServiceContext.from_defaults(
        llm=llm,
        embed_model=embed_model,
        node_parser=node_parser,
    )
    if not os.path.exists(save_dir):
        sentence_index = VectorStoreIndex.from_documents(
            documents, service_context=sentence_context
        )
        sentence_index.storage_context.persist(persist_dir=save_dir)
    else:
        sentence_index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=save_dir),
            service_context=sentence_context,
        )

    return sentence_index

def get_sentence_window_query_engine(
        sentence_index,
        similarity_top_k=6,
        rerank_top_n=2,
):
    postproc = MetadataReplacementPostProcessor(target_metadata_key="window")
    rerank = SentenceTransformerRerank(top_n=rerank_top_n, model="BAAI/bge-reranker-base"
                                       )
    sentence_window_engine = sentence_index.as_query_engine(
        similarity_top_k=similarity_top_k, node_postprocessors=[postproc, rerank]
    )
    return sentence_window_engine