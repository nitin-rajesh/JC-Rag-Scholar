import React from 'react';
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import ChatApp from './pages/ChatApp';
import {
  ReactiveBase,
} from "@appbaseio/reactivesearch";
import SingleList from "@appbaseio/reactivesearch/lib/components/list/SingleList";
import ReactiveList from "@appbaseio/reactivesearch/lib/components/result/ReactiveList";
import ResultList from "@appbaseio/reactivesearch/lib/components/result/ResultList";

function App() {
  return (
      <Router>
        <Routes>
          <Route path="/" element={<ChatApp />} />
        </Routes>
      </Router>
  );
}

export default App;
