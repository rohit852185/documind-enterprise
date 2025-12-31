from src.rag.rag_chain import run_rag



def ask():
    print("\n🟦 DocuMind Enterprise RAG System")
    print("----------------------------------\n")

    while True:
        query = input("Ask a question (or type 'exit'): ")

        if query.lower() == "exit":
            break

        print("\nSearching...\n")
        answer = run_rag(query)


        print("\nANSWER:\n")
        print(answer)
        print("\n----------------------------------\n")


if __name__ == "__main__":
    ask()
