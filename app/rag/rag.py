import logging
import os
from dotenv import load_dotenv
from groq import Groq
from db.conection_qdrant import search_similar_documents, create_collection_if_not_exists
from cache.cag import get_cached_response, set_cached_response
from embeddings.embeddings import generate_embeddings_from_context_file
from config.config import (CONTEXT_DADOS_ESCOLA, CONTEXT_DATAS_VACINAS, CONTEXT_DICIONARIO_VACINAS,
                           CONTEXT_ESCOLAS_MUNICIPAIS, CONTEXT_FAIXAS_TRANSPORTE, CONTEXT_LOCAIS_POSTOS,
                           CONTEXT_POSTOS_VACINA, CONTEXT_TRANSPORTE, CONTEXT_NOVA_BASE)

logging.basicConfig(level=logging.DEBUG)

load_dotenv()

api_key = os.environ.get('GROQ_API_KEY')
if not api_key:
    raise ValueError("Erro: 'GROQ_API_KEY' não encontrada no ambiente.")
client = Groq(api_key=api_key)

create_collection_if_not_exists()

PROMPT_TEMPLATE = """
"Você é a Aurora, uma IA especializada em fornecer informações sobre a cidade de Recife em educação,saúde e transporte. Seu conhecimento se limita exclusivamente aos dados contidos nos arquivos JSON fornecidos..."

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
    if not isinstance(similar_documents, list):  # Verifica se é uma lista válida
        logging.error(f"Formato inesperado para similar_documents: {type(similar_documents)} | Valor: {similar_documents}")
        return []

    valid_documents = [doc for doc in similar_documents if isinstance(doc, dict)]
    
    if not valid_documents:
        logging.error("Nenhum documento válido encontrado.")
        return []

    sorted_documents = sorted(valid_documents, key=lambda x: x.get('score', 0), reverse=True)
    logging.debug(f"Top {len(sorted_documents)} documentos selecionados por pontuação.")
    return sorted_documents[:max_documents]

def get_rag_response(prompt: str, similar_documents):
    contexts = [ CONTEXT_DADOS_ESCOLA, CONTEXT_DATAS_VACINAS, CONTEXT_DICIONARIO_VACINAS, 
    CONTEXT_ESCOLAS_MUNICIPAIS, CONTEXT_FAIXAS_TRANSPORTE, CONTEXT_LOCAIS_POSTOS, 
    CONTEXT_POSTOS_VACINA, CONTEXT_TRANSPORTE, CONTEXT_NOVA_BASE]
    
    # Embeddings para saúde e educação
    health_and_education_contexts = [
        CONTEXT_DADOS_ESCOLA,
        CONTEXT_DATAS_VACINAS,
        CONTEXT_DICIONARIO_VACINAS,
        CONTEXT_ESCOLAS_MUNICIPAIS,
        CONTEXT_POSTOS_VACINA
    ]
    
    # Função que gera embeddings de todos os contextos
    def generate_all_embeddings(contexts):
        embeddings = []
        for context in contexts:
            try:
                embeddings_from_context = generate_embeddings_from_context_file(context)
                embeddings.extend(embeddings_from_context)
                logging.debug(f"Embeddings gerados para o contexto {context}: {len(embeddings_from_context)} registros")
            except Exception as e:
                logging.error(f"Erro ao gerar embeddings para o contexto {context}: {e}")
        return embeddings
    
    embeddings = generate_all_embeddings(contexts)
    health_and_education_embeddings = generate_all_embeddings(health_and_education_contexts)
    
    embeddings.extend(health_and_education_embeddings)
    
    logging.debug(f"Total de embeddings gerados: {len(embeddings)} registros")
    
    cached_response = get_cached_response(prompt)
    if cached_response:
        logging.debug(f"Resposta em cache encontrada para o prompt: {prompt}")
        return cached_response
    
    logging.debug(f"Buscando documentos semelhantes para o prompt: {prompt}")

    if not isinstance(similar_documents, list):
        logging.error(f"Erro: similar_documents não é uma lista. Tipo recebido: {type(similar_documents)} | Valor: {similar_documents}")
        similar_documents = []  
    
    top_documents = select_top_documents(similar_documents)
    logging.debug(f"Top {len(top_documents)} documentos selecionados para contexto")

    context = "\n\n".join(embeddings) if embeddings else ""
    logging.debug(f"Contexto inicial gerado com {len(context)} caracteres.")
    
    if top_documents:
        context += "\n\nAh, encontrei algumas informações que podem ser úteis:\n"
        for doc in top_documents:
            if isinstance(doc, dict):
                content = doc.get('content', 'Sem conteúdo disponível')
                context += f"• {content}\n"
            else:
                logging.warning(f"Documento inesperado encontrado: {doc}")
                content = "Erro ao processar documento."
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

        if not chat_completion or not hasattr(chat_completion, 'choices') or not chat_completion.choices:
            logging.error("Resposta inválida da API do Groq.")
            return "Erro ao gerar a resposta. Por favor, tente novamente."

        response = chat_completion.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"Erro ao chamar a API do Groq: {e}")
        return "Desculpe, ocorreu um erro ao gerar a resposta."
    
    logging.debug(f"Resposta gerada pelo modelo: {response}")

    if not response or response.startswith(".\n"):
        logging.debug("Resposta vazia ou inválida detectada.")
        response = "Desculpe, acho que algo deu errado! Pode repetir sua dúvida?"
    
    set_cached_response(prompt, response)
    return response
