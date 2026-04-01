import React, { useState } from "react";
import "./App.css";

function App() {
  const [query, setQuery] = useState("");
  const [answer, setAnswer] = useState("");
  const [sources, setSources] = useState([]);
  const [loading, setLoading] = useState(false);

  const askQuestion = async () => {
    if (!query.trim()) return;

    setLoading(true);
    setAnswer("");
    setSources([]);

    try {
      const res = await fetch("http://127.0.0.1:8000/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query }),
      });

      const data = await res.json();
      setAnswer(data.answer);
      setSources(data.sources || []);
    } catch (err) {
      setAnswer("Error connecting to backend.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container">
      <h1>🧬 Cancer Research RAG Assistant</h1>
      <p className="subtitle">
        Evidence-based answers from PubMed & medical guidelines
      </p>

      <textarea
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        placeholder="Ask a medical research question..."
      />

      <button onClick={askQuestion} disabled={loading}>
        {loading ? "Thinking..." : "Ask"}
      </button>

      {answer && (
        <div className="card">
          <h2>Answer</h2>
          <p>{answer}</p>
        </div>
      )}

      {sources.length > 0 && (
        <div className="card">
          <h2>Sources</h2>
          {sources.map((src, idx) => (
            <div key={idx} className="source">
              <strong>Source {idx + 1}</strong>
              <p>{src.content}</p>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default App;
