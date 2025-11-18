import axios from 'axios';

const logApiEvent = async (data) => {
  console.log("Logging API event:", data); // Add this line
  try {
    await axios.post('http://localhost:5001/api/log', data);
  } catch (error) {
    console.error('Logging failed:', error);
  }
};

export default logApiEvent;
