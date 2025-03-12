import logging
import os
from dotenv import load_dotenv
from groq import Groq
from db.conection_qdrant import search_similar_documents, create_collection_if_not_exists
from cache.cag import get_cached_response, set_cached_response
from embeddings.embeddings import generate_embeddings_from_context_file
from config.config import CONTEXT_DADOS_ESCOLA
from config.config import CONTEXT_DATAS_VACINAS
from config.config import CONTEXT_DICIONARIO_VACINAS
from config.config import CONTEXT_ESCOLAS_MUNICIPAIS
from config.config import CONTEXT_FAIXAS_TRANSPORTE
from config.config import CONTEXT_LOCAIS_POSTOS
from config.config import CONTEXT_POSTOS_VACINA
from config.config import CONTEXT_TRANSPORTE


logging.basicConfig(level=logging.DEBUG)

load_dotenv()

api_key = os.environ.get('GROQ_API_KEY')
if not api_key:
    raise ValueError("Erro: 'GROQ_API_KEY' não encontrada no ambiente.")
client = Groq(api_key=api_key)

create_collection_if_not_exists()

PROMPT_TEMPLATE = """
Você é a Aurora, uma IA especializada em fornecer informações sobre a cidade de Recife. Seu conhecimento se limita exclusivamente aos dados contidos nos arquivos JSON fornecidos. Sempre que for responder, siga estas diretrizes
Contexto relevante:
{context}

Regras para resposta:
1- Responda apenas com base nos dados disponíveis nos arquivos JSON. Se a informação solicitada não estiver presente, diga algo como: ‘Desculpe, não encontrei essa informação nos meus arquivos.’

2- Se não houver informações suficientes no contexto, responda:  
"Desculpe, não encontrei informações suficientes para responder à sua pergunta. Mas estou aqui para ajudar no que for possível!"

3- Nunca invente ou forneça informações externas.  
NÃO mencione explicitamente que está seguindo um contexto na resposta final.

4- Seja objetiva e clara, apresentando as informações de forma acessível para qualquer usuário.

5- Mantenha a formalidade e a precisão, especialmente nos tópicos de saúde, educação e transporte.

### Pergunta:
{question}

Dica: Sempre forneça a resposta mais completa possível, respeitando as diretrizes acima.
"""

def select_top_documents(similar_documents, max_documents=10):
    if not similar_documents:
        return []

    if isinstance(similar_documents[0], dict):
        sorted_documents = sorted(similar_documents, key=lambda x: x.get('score', 0), reverse=True)
    else:
        sorted_documents = similar_documents[:max_documents]
    
    return sorted_documents[:max_documents]

def get_rag_response(prompt: str, similar_documents):
    context_escola_dados = CONTEXT_DADOS_ESCOLA
    context_vacinas_datas = CONTEXT_DATAS_VACINAS
    context_vacinas_dicionarios = CONTEXT_DICIONARIO_VACINAS
    context_municipais_escolas = CONTEXT_ESCOLAS_MUNICIPAIS
    context_transporte_faixa = CONTEXT_FAIXAS_TRANSPORTE
    context_postos_locais = CONTEXT_LOCAIS_POSTOS
    context_vacinas_postos = CONTEXT_POSTOS_VACINA
    context_veiculo = CONTEXT_TRANSPORTE

    embeddings = generate_embeddings_from_context_file(context_escola_dados)
    embeddings = generate_embeddings_from_context_file(context_vacinas_datas)
    embeddings = generate_embeddings_from_context_file(context_vacinas_dicionarios)
    embeddings = generate_embeddings_from_context_file(context_municipais_escolas)
    embeddings = generate_embeddings_from_context_file(context_transporte_faixa)
    embeddings = generate_embeddings_from_context_file(context_postos_locais)
    embeddings = generate_embeddings_from_context_file(context_vacinas_postos)
    embeddings = generate_embeddings_from_context_file(context_veiculo)
    logging.debug(f"Embeddings gerados: {len(embeddings)} registros")
    
    cached_response = get_cached_response(prompt)
    if cached_response:
        logging.debug(f"Resposta em cache encontrada para o prompt: {prompt}")
        return cached_response

    logging.debug(f"Buscando documentos semelhantes para o prompt: {prompt}")
    search_results = search_similar_documents(prompt)
    logging.debug(f"Documentos semelhantes encontrados: {len(search_results) if search_results else 0}")

    similar_documents = search_results if search_results else []
    top_documents = select_top_documents(similar_documents)
    logging.debug(f"Top {len(top_documents)} documentos selecionados para contexto")

    context = "\n\n".join(embeddings) if embeddings else ""
    
    if top_documents:
        context += "\n\nAh, encontrei algumas informações que podem ser úteis:\n"
        for doc in top_documents:
            content = doc.get('content', 'Sem conteúdo disponível') if isinstance(doc, dict) else doc
            context += f"• {content}\n"
    
    max_context_length = 2000
    if len(context) > max_context_length:
        logging.debug("Contexto muito grande, truncando para 2000 caracteres...")
        context = context[:max_context_length] + "...\nSe quiser mais detalhes, me avise!"
    
    if not context.strip():
        logging.debug("Nenhum contexto encontrado para a resposta.")
        return "Desculpe, não encontrei informações suficientes para responder sua pergunta. Se quiser reformular, estou aqui para ajudar!"

    messages = [
        {"role": "system", "content": "Você é Aurora, uma IA especializada em fornecer informações sobre a cidade de Recife. Responda com base no contexto, construa a melhor resposta com uma abordagem clara para o usuário. Lembre-se de sempre consultar o contexto antes de responder. Não mencione o contexto na saída final."},
        {"role": "user", "content": PROMPT_TEMPLATE.format(context=context, question=prompt)}
    ]
    
    try:
        logging.debug("Enviando requisição para a API do Groq...")
        chat_completion = client.chat.completions.create(messages=messages, model="llama-3.3-70b-versatile")
    except Exception as e:
        logging.error(f"Erro ao chamar a API do Groq: {e}")
        return "Desculpe, ocorreu um erro ao gerar a resposta."

    if not chat_completion.choices:
        logging.debug("Nenhuma escolha retornada pela API do Groq.")
        return "Desculpe, não consegui gerar uma resposta."
    
    response = chat_completion.choices[0].message.content.strip()
    logging.debug(f"Resposta gerada pelo modelo: {response}")
    
    if not response or response.startswith(".\n"):
        logging.debug("Resposta vazia ou inválida detectada.")
        response = "Desculpe, acho que algo deu errado! Pode repetir sua dúvida?"

    set_cached_response(prompt, response)
    return response