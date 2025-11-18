import React, { useState } from "react";
import axios from 'axios';
import logApiEvent from "../services/axiosLogger";

import {
  ReactiveBase,
} from "@appbaseio/reactivesearch";

const ChatApp = () => {
  const [messages, setMessages] = useState([
    { sender: "bot", text: "Hi there" }
  ]);
  const [input, setInput] = useState("");

  const handleSend = async () => {
    if (input.trim() === "") return;

    const userMessage = { sender: "user", text: input };
    const typingMessage = { sender: "bot", text: "...", id: "typing" }; // Add ID to identify it

    setMessages((prev) => [...prev, userMessage, typingMessage]);
    setInput("");

    try {
      const response = await askQuestion(input);
      const botText = response || "Sorry, I couldn't understand that.";
      const botMessage = { sender: "bot", text: botText };

      // Replace the typing message
      setMessages((prev) =>
        prev.map((msg) => (msg.id === "typing" ? botMessage : msg))
      );
    } catch (err) {
      const errorMessage = { sender: "bot", text: "Error getting response." };
      setMessages((prev) =>
        prev.map((msg) => (msg.id === "typing" ? errorMessage : msg))
      );
    }
  };



  const dummyResponse = (msg) => {
    return "I need to think about it";
  };

  const askQuestion = async (question) => {
    const start = performance.now();
    try {
      const response = await axios.post('http://localhost:8000/ask', {
        question,
      }, {
        headers: {
          'Content-Type': 'application/json',
        },
      });
      const duration = performance.now() - start;
      logApiEvent({
        type: 'api-call',
        endpoint: '/api/send',
        status: 'success',
        duration,
      });
      const parsed = JSON.parse(response.data);

      return parsed.answer;

    } catch (error) {
      console.error('Error:', error);
      const duration = performance.now() - start;

      logApiEvent({
        type: 'api-call',
        endpoint: '/api/send',
        status: 'error',
        duration,
        error: error.message,
      });
      return dummyResponse('');
    }

  };

  const handleKeyPress = (e) => {
    if (e.key === "Enter") handleSend();
  };

  return (
    <div
      className="w-full min-h-screen bg-cover bg-center flex flex-col items-center justify-center"
      style={{ backgroundImage: "url('https://www.thoughtco.com/thmb/pTpEDJZxgSg0-t4yk8UFkauifkU=/1500x0/filters:no_upscale():max_bytes(150000):strip_icc()/51246880-resize-56a85e953df78cf7729dcc17.jpg')" }}
    >
      <div className="w-[100px] sm:w-[400px] md:w-[500px] max-w-full bg-white bg-opacity-95 rounded-2xl shadow-lg p-4 flex flex-col h-[95vh]">
        <img
          alt="logo hai bhai idhar"
          src="https://img.icons8.com/?size=192&id=9IGyQOZMnPtv&format=png"
          className="w-12 h-12 mb-4 object-contain mx-auto"
        />
        <div className="flex-1 overflow-y-auto mb-4 flex flex-col space-y-2">
          {messages.map((msg, index) => (
            <div
              key={index}
              className={`w-full flex ${msg.sender === "user" ? "justify-end" : "justify-start"}`}
            >
              <div
                className={`px-4 py-2 rounded-2xl text-sm break-words ${
                  msg.sender === "user"
                    ? "bg-blue-400 text-white"
                    : "bg-gray-200 text-black"
                } max-w-[75%]`}
              >

                <div className="mb-1 font-semibold">
                  {msg.sender === "user" ? "User:" : "Bot:"}
                </div>
                <div className="whitespace-pre-wrap">{msg.text}</div>
              </div>
            </div>
          ))}
        </div>

        <div className="w-full flex">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyUp={handleKeyPress}
            placeholder="Type a message..."
            className="flex-1 p-2 text-xs rounded-l-full border border-gray-300 focus:outline-none"
          />
          <button
            onClick={handleSend}
            className="bg-blue-500 text-xs text-white px-4 rounded-r-full hover:bg-blue-600"
          >
            Send
          </button>
        </div>
        <div className="text-center text-xs italic mt-3 text-gray-500">*For preview only. Verify information from official site.</div>

      </div>
    </div>
  );
};

export default ChatApp;