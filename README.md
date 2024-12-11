# Open WebUI Collections

A collection of custom tools and extensions for Open WebUI that enhance its capabilities through advanced features like RAG (Retrieval-Augmented Generation). This repository provides ready-to-use components for implementing memory management and context-aware responses using Pinecone vector database, allowing you to build more intelligent and context-aware chatbot applications.

## Support the Project

[![PayPal](https://img.shields.io/badge/PayPal-00457C?style=for-the-badge&logo=paypal&logoColor=white)](https://www.paypal.com/paypalme/aahadr)

## Currently Available Collections

### Memory Management Tools with Pinecone

1. **Add to Memory (Pinecone)**
   - File: `filters/add_to_memory_pinecone.py`
   - Purpose: Enables storing information into Pinecone vector database for later retrieval
   - Integration: Can be used as a custom tool in Open WebUI

2. **Memory Filter (Pinecone)**
   - File: `filters/memory_filter_pinecone.py`
   - Purpose: Retrieves stored information from Pinecone based on context similarity
   - Integration: Implements custom RAG (Retrieval-Augmented Generation) functionality

### Web Interaction Tools

1. **Better Web Scrape**
   - File: `tools/better_web_scrape.py`
   - Purpose: Enhanced web scraping functionality using Jina service
   - Integration: Enables efficient extraction of web content for processing

2. **YouTube Transcript Tools**
   - File: `tools/youtube_transcript_tools.py`
   - Purpose: Extract and process YouTube video transcripts
   - Integration: Allows working with YouTube video content through transcripts

## Understanding RAG (Retrieval-Augmented Generation)

RAG (Retrieval-Augmented Generation) is a powerful approach that combines the capabilities of large language models with the ability to access and utilize external knowledge bases. By implementing RAG, you can enhance your AI applications with accurate, up-to-date, and contextually relevant information, while maintaining control over the knowledge that influences the model's responses.

The key advantage of RAG lies in its ability to bridge the gap between static model knowledge and dynamic, customized information needs. When integrated with vector databases like Pinecone, RAG enables efficient similarity-based searches, allowing the system to retrieve and incorporate the most relevant information into its responses. This not only improves the accuracy and reliability of the AI's outputs but also helps in reducing hallucinations and providing verifiable information based on your stored knowledge.

## Getting Started

1. Clone this repository
2. Choose the tools you want to integrate with your Open WebUI installation
3. Follow the specific documentation for each tool in their respective directories

## Contributing

Feel free to contribute by adding new tools, improving existing ones, or enhancing the documentation. Pull requests are welcome!

## License

MIT License
