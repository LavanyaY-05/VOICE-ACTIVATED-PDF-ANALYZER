<html>
  <body>
    <h2>🔊 VOICE ACTIVATED PDF ANALYZER (VAPA)</h2>
<h4>📌 ABOUT THE PROJECT </h4>
<p>
VAPA (Voice Activated PDF Analyzer) is a web application that generates answer to the user queries based on the content of the uploaded PDF. The feature added to the web application is speech-to-speech technology where users can ask question either through voice or text input. The system responds with answers in both voice and text formats, enabling a complete speech-to-speech interaction experience.</p>
<p> 
This application is built using LangChain and integrates a set of NLP tools and AI models to extract, understand, and respond based on the content of the PDF file.</p>
<h4>⚙️ Technologies and Tools </h4>
<ol>
  <li><strong> PDF.js </strong> – Extracts text content from uploaded PDF files in the frontend</li>
  <li><strong> LangChain </strong> – Coordinates the document pipeline: loading, chunking, embedding, and querying</li>
  <li><strong> Sentence Transformers </strong> – Converts text into embeddings for semantic similarity (using <code>all-MiniLM-L6-v2</code>)</li>
  <li><strong> FAISS </strong> (Facebook AI Similarity Search) – Indexes and retrieves the most relevant document chunks using vector similarity</li>
  <li><strong> Web Speech API </strong> – Enables speech recognition and text-to-speech synthesis in the browser for seamless voice interaction</li>
</ol>
<h4> How it Works</h4>
<ol>
  <li>User uploads a PDF file and ask query to the application</li>
  <li>The backend processes the PDF using LangChain and stores it in a searchable format using FAISS</li>
  <li>The user can ask a question via voice or text</li>
  <li>The system retrieves the most relevant content from the document</li>
  <li>The response is spoken aloud and displayed as text</li>
</ol>

  </body>
</html>




