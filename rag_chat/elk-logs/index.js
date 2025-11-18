const express = require('express');
const cors = require('cors');
const { Client } = require('@elastic/elasticsearch');

const app = express();
app.use(cors());
app.use(express.json());

// Elasticsearch client
const esClient = new Client({ node: 'http://localhost:9200' });

app.post('/api/log', async (req, res) => {
  try {
    const logData = req.body;

    await esClient.index({
      index: 'chat-logs',
      type: '_doc', // Required for ES 6.x
      body: logData,
    });

    res.status(200).send('Log saved');
    console.log('Log saved: ', logData);
  } catch (err) {
    console.error(err);
    res.status(500).send('Error saving log');
    console.log('Error saving log');

  }
});

app.listen(5001, () => console.log('Logging server on http://localhost:5001'));
