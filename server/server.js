/**
 * Hidden Connections - Embedding API Server
 * Provides live embedding generation via OpenAI API
 *
 * Usage:
 *   npm install
 *   OPENAI_API_KEY=sk-... node server.js
 *
 * Endpoints:
 *   POST /api/embed - Generate embedding for text
 *   POST /api/embed-batch - Generate embeddings for multiple texts
 */

import 'dotenv/config';
import express from 'express';
import cors from 'cors';
import OpenAI from 'openai';

const app = express();

// Middleware
app.use(cors());
app.use(express.json());

// Configuration
const CONFIG = {
  model: 'text-embedding-3-large',
  dimensions: 256,  // Matryoshka: 256 dims still outperforms ada-002 at 1536
  port: process.env.PORT || 3001,
};

// Initialize OpenAI client
const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

// Validate API key on startup
if (!process.env.OPENAI_API_KEY) {
  console.error('ERROR: OPENAI_API_KEY environment variable is required');
  console.error('Set it with: export OPENAI_API_KEY=sk-...');
  process.exit(1);
}

/**
 * POST /api/embed
 * Generate embedding for a single text
 *
 * Request body: { text: string }
 * Response: { embedding: number[], dimensions: number }
 */
app.post('/api/embed', async (req, res) => {
  try {
    const { text } = req.body;

    if (!text || typeof text !== 'string') {
      return res.status(400).json({
        error: 'Missing or invalid text parameter',
        message: 'Request body must include { text: "your text here" }'
      });
    }

    if (text.trim().length === 0) {
      return res.status(400).json({
        error: 'Empty text',
        message: 'Text cannot be empty'
      });
    }

    console.log(`Embedding request: "${text.substring(0, 50)}${text.length > 50 ? '...' : ''}"`);

    const response = await openai.embeddings.create({
      model: CONFIG.model,
      input: text,
      dimensions: CONFIG.dimensions,
    });

    const embedding = response.data[0].embedding;

    res.json({
      embedding,
      dimensions: CONFIG.dimensions,
      model: CONFIG.model,
      usage: response.usage,
    });

  } catch (err) {
    console.error('Embedding error:', err.message);

    if (err.code === 'invalid_api_key') {
      return res.status(401).json({ error: 'Invalid OpenAI API key' });
    }
    if (err.code === 'rate_limit_exceeded') {
      return res.status(429).json({ error: 'Rate limit exceeded, please try again' });
    }

    res.status(500).json({
      error: 'Embedding generation failed',
      message: err.message
    });
  }
});

/**
 * POST /api/embed-batch
 * Generate embeddings for multiple texts
 *
 * Request body: { texts: string[] }
 * Response: { embeddings: number[][], dimensions: number }
 */
app.post('/api/embed-batch', async (req, res) => {
  try {
    const { texts } = req.body;

    if (!Array.isArray(texts) || texts.length === 0) {
      return res.status(400).json({
        error: 'Missing or invalid texts parameter',
        message: 'Request body must include { texts: ["text1", "text2", ...] }'
      });
    }

    // Validate all texts
    const invalidIdx = texts.findIndex(t => typeof t !== 'string' || t.trim().length === 0);
    if (invalidIdx !== -1) {
      return res.status(400).json({
        error: `Invalid text at index ${invalidIdx}`,
        message: 'All texts must be non-empty strings'
      });
    }

    // Limit batch size
    if (texts.length > 100) {
      return res.status(400).json({
        error: 'Batch too large',
        message: 'Maximum 100 texts per batch request'
      });
    }

    console.log(`Batch embedding request: ${texts.length} texts`);

    const response = await openai.embeddings.create({
      model: CONFIG.model,
      input: texts,
      dimensions: CONFIG.dimensions,
    });

    const embeddings = response.data.map(item => item.embedding);

    res.json({
      embeddings,
      dimensions: CONFIG.dimensions,
      model: CONFIG.model,
      count: embeddings.length,
      usage: response.usage,
    });

  } catch (err) {
    console.error('Batch embedding error:', err.message);
    res.status(500).json({
      error: 'Batch embedding generation failed',
      message: err.message
    });
  }
});

/**
 * GET /api/health
 * Health check endpoint
 */
app.get('/api/health', (req, res) => {
  res.json({
    status: 'ok',
    model: CONFIG.model,
    dimensions: CONFIG.dimensions,
  });
});

/**
 * GET /api/config
 * Get current embedding configuration
 */
app.get('/api/config', (req, res) => {
  res.json({
    model: CONFIG.model,
    dimensions: CONFIG.dimensions,
    description: 'OpenAI text-embedding-3-large with Matryoshka dimensionality reduction',
  });
});

// Start server
app.listen(CONFIG.port, () => {
  console.log('');
  console.log('╔════════════════════════════════════════════════════════════╗');
  console.log('║       Hidden Connections - Embedding API Server            ║');
  console.log('╠════════════════════════════════════════════════════════════╣');
  console.log(`║  Model: ${CONFIG.model.padEnd(42)}║`);
  console.log(`║  Dimensions: ${String(CONFIG.dimensions).padEnd(38)}║`);
  console.log(`║  Port: ${String(CONFIG.port).padEnd(43)}║`);
  console.log('╠════════════════════════════════════════════════════════════╣');
  console.log('║  Endpoints:                                                ║');
  console.log('║    POST /api/embed       - Single text embedding           ║');
  console.log('║    POST /api/embed-batch - Batch text embeddings           ║');
  console.log('║    GET  /api/health      - Health check                    ║');
  console.log('║    GET  /api/config      - Get configuration               ║');
  console.log('╚════════════════════════════════════════════════════════════╝');
  console.log('');
  console.log(`Server running at http://localhost:${CONFIG.port}`);
});
