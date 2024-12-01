import os
import sys
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
import warnings
import subprocess

vectorstore = None
os.environ['TOKENIZERS_PARALLELISM'] = 'true'
os.environ['GROQ_API_KEY'] = input("GROQ_API_KEY: ")
warnings.filterwarnings("ignore", message="cumsum_out_mps supported by MPS on MacOS 13+")

def get_paths_from_filesystem(root_dir):
    all_paths = []
    if os.access(root_dir, os.R_OK):
        try:
            for entry in os.scandir(root_dir):
                all_paths.append(entry.path)
                if entry.is_dir(follow_symlinks=False):
                    all_paths.extend(get_paths_from_filesystem(entry.path))
        except PermissionError:
            pass
        except Exception as e:
            pass
    return all_paths

def read_paths_from_file(file_path):
    try:
        with open(file_path, "r") as file:
            paths = file.read().splitlines()
        return paths
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        sys.exit(1)

def write_paths_to_file(paths, file_path):
    try:
        with open(file_path, "w") as file:
            for path in sorted(paths):
                file.write(path + "\n")
        #print(f"Updated {file_path} with new paths.")
    except Exception as e:
        print(f"Error writing to {file_path}: {e}")
        sys.exit(1)

def update_directories_and_files(file_paths, root_directory):
    current_paths = get_paths_from_filesystem(root_directory)
    
    if current_paths != file_paths:
        write_paths_to_file(current_paths, "files/directories.txt")
        file_paths = read_paths_from_file("files/directories.txt")
        #print("The directories and files have been updated.")
        return True, file_paths
    else:
        #print("No changes detected in the directories and files.")
        return False, file_paths

def load_and_split_text(text_file_path):
    try:
        loader = TextLoader(text_file_path)
        documents = loader.load()
        #print(f'Loaded {len(documents)} documents')
    except Exception as e:
        print(f"Error loading the text file: {e}")
        sys.exit(1)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)
    #print(f"Created {len(splits)} splits from the text file")
    return splits

def setup_vectorstore(splits):
    global vectorstore
    vectorstore = Chroma.from_documents(documents=splits, embedding=HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2'))
    retriever = vectorstore.as_retriever(search_kwargs={"k": 1})
    return retriever

def get_prompts():
    try:
        with open("files/sys_prompt.txt", "r") as file:
            system_prompt = file.read().strip()
    except Exception as e:
        print(f"Error reading sys_prompt.txt: {e}")
        sys.exit(1)
        
    question_prompt_template = """Current context and question:

{context}

Question: {question}

Answer:"""
    return system_prompt, question_prompt_template

def setup_rag_chain(retriever, system_prompt, question_prompt_template):
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", question_prompt_template),
    ])

    llm = ChatGroq(model = "llama3-70b-8192",
                    temperature = 0.2,
                    max_tokens=4096,
                    stream=False,
                    timeout=None,
                    max_retries=2)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {
            "context": lambda x: format_docs(retriever.invoke(x["question"])),
            "question": lambda x: x["question"]
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain, format_docs

def run_bash_command(command):
    try:
        powershell_command = ['powershell', '-NoProfile', '-NonInteractive', '-Command', command]
        process = subprocess.run(
            powershell_command,
            capture_output=True,
            text=True,
            check=False
        )

        stdout, stderr = process.stdout, process.stderr

        return stdout, stderr

    except Exception as e:
        return None, str(e)


def main():
    global vectorstore

    print("Getting paths...")

    root_directory = os.path.join(os.getcwd(), "rootdir")

    text_file_path = "files/directories.txt"
    file_paths = read_paths_from_file(text_file_path)
    updated, file_paths = update_directories_and_files(file_paths, root_directory)

    print("Loading paths...")

    splits = load_and_split_text(text_file_path)
    retriever = setup_vectorstore(splits)
    system_prompt, question_prompt_template = get_prompts()
    rag_chain, format_docs = setup_rag_chain(retriever, system_prompt, question_prompt_template)

    os.system("cls")
    print("BashGen 0.1")
    print("'quit' to exit")
    print("'clr' to clear screen")
    print()

    updated = False

    while True:
        user_question = input(">>")

        if user_question.strip() == "":
            continue

        if user_question.lower() == 'quit':
            print("Goodbye!")
            break

        if user_question.lower() == 'clr':
            os.system("cls")
            print("BashGen 0.1")
            print("'quit' to exit")
            print("'clr' to clear screen")
            print()
            continue

        try:
            # Retrieve context
            context = format_docs(retriever.invoke(user_question))
            
            # Get result from RAG chain
            result = rag_chain.invoke({"question": user_question, "context": context})

            bash_command = result.strip()
            
            
            if bash_command:
                print(f"{bash_command}")
                #output, error = run_bash_command(bash_command)
                if output:
                    print("\nOutput:")
                    print(output)
                if error:
                    print("\nError:")
                    print(error)
                print()

            # Log the result
            with open("files/.history", "a") as f:
                f.write(f"{user_question}: {result}")
                f.write("\n")

            updated, file_paths = update_directories_and_files(file_paths, root_directory)
            if updated:
                print("\nUpdating...")
                # If updated, reload and reinitialize the RAG chain
                vectorstore.delete_collection()
                splits = load_and_split_text(text_file_path)
                retriever = setup_vectorstore(splits)
                rag_chain, format_docs = setup_rag_chain(retriever, system_prompt, question_prompt_template)
                print("Updated paths\n")

        except Exception as e:
            print(f"An error occurred while processing your question: {e}")
            input("\nPress Enter to continue...")

if __name__ == "__main__":
    main()
