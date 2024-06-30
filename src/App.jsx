import { useState, useEffect } from 'react';
import './App.css';
import axios from 'axios';

function App() {
  const [content, setContent] = useState('');

  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await axios.get('http://localhost:3000/api/content');
        console.log(response)
        setContent(response.data[0].content); // Assuming response.data is the content fetched from API
      } catch (error) {
        console.error('Error fetching content:', error);
      }
    };

    fetchData();
  }, []); // Empty dependency array ensures this effect runs only once on component mount

  return (
    <div className="App">
      <h1>Content from API</h1>
      <div className="content" dangerouslySetInnerHTML={{ __html: content }} />
    </div>
  );
}

export default App;
