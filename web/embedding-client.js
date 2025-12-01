/**
 * Hidden Connections - Live Embedding Client
 *
 * Provides functions to:
 * - Get embeddings from the API server
 * - Find nearest neighbors in the existing point cloud
 * - Add new points to the visualization dynamically
 *
 * Usage:
 *   import { EmbeddingClient } from './embedding-client.js';
 *   const client = new EmbeddingClient('http://localhost:3001');
 *   const embedding = await client.getEmbedding('my text');
 */

export class EmbeddingClient {
    constructor(apiUrl = 'http://localhost:3001') {
        this.apiUrl = apiUrl.replace(/\/$/, ''); // Remove trailing slash
        this.dimensions = 256;
        this.isAvailable = false;

        // Check API availability on init
        this.checkHealth();
    }

    /**
     * Check if the API server is available
     */
    async checkHealth() {
        try {
            const res = await fetch(`${this.apiUrl}/api/health`);
            if (res.ok) {
                const data = await res.json();
                this.dimensions = data.dimensions;
                this.isAvailable = true;
                console.log(`Embedding API connected: ${data.model} (${data.dimensions}d)`);
                return true;
            }
        } catch (err) {
            console.warn('Embedding API not available:', err.message);
            this.isAvailable = false;
        }
        return false;
    }

    /**
     * Get embedding for a single text
     * @param {string} text - Text to embed
     * @returns {Promise<number[]>} - Embedding vector
     */
    async getEmbedding(text) {
        if (!text || typeof text !== 'string') {
            throw new Error('Text must be a non-empty string');
        }

        const res = await fetch(`${this.apiUrl}/api/embed`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text }),
        });

        if (!res.ok) {
            const error = await res.json();
            throw new Error(error.message || 'Embedding request failed');
        }

        const data = await res.json();
        return data.embedding;
    }

    /**
     * Get embeddings for multiple texts
     * @param {string[]} texts - Array of texts to embed
     * @returns {Promise<number[][]>} - Array of embedding vectors
     */
    async getEmbeddingsBatch(texts) {
        if (!Array.isArray(texts) || texts.length === 0) {
            throw new Error('Texts must be a non-empty array');
        }

        const res = await fetch(`${this.apiUrl}/api/embed-batch`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ texts }),
        });

        if (!res.ok) {
            const error = await res.json();
            throw new Error(error.message || 'Batch embedding request failed');
        }

        const data = await res.json();
        return data.embeddings;
    }

    /**
     * Calculate cosine similarity between two vectors
     * @param {number[]} a - First vector
     * @param {number[]} b - Second vector
     * @returns {number} - Cosine similarity (0 to 1)
     */
    cosineSimilarity(a, b) {
        if (a.length !== b.length) {
            throw new Error('Vectors must have same length');
        }

        let dotProduct = 0;
        let normA = 0;
        let normB = 0;

        for (let i = 0; i < a.length; i++) {
            dotProduct += a[i] * b[i];
            normA += a[i] * a[i];
            normB += b[i] * b[i];
        }

        const denominator = Math.sqrt(normA) * Math.sqrt(normB);
        return denominator > 0 ? dotProduct / denominator : 0;
    }

    /**
     * Find k nearest neighbors from a set of pre-computed embeddings
     * @param {number[]} queryEmbedding - Query embedding vector
     * @param {Object[]} points - Array of points with embeddings
     * @param {number} k - Number of neighbors to find
     * @returns {Object[]} - Array of {point, similarity} sorted by similarity
     */
    findNearestNeighbors(queryEmbedding, points, k = 5) {
        const similarities = points
            .filter(p => p.embedding) // Only points with embeddings
            .map(point => ({
                point,
                similarity: this.cosineSimilarity(queryEmbedding, point.embedding),
            }))
            .sort((a, b) => b.similarity - a.similarity);

        return similarities.slice(0, k);
    }

    /**
     * Estimate 2D position for a new embedding using weighted average
     * of nearest neighbors' positions (simple projection approximation)
     *
     * @param {number[]} embedding - New embedding vector
     * @param {Object[]} points - Existing points with embeddings and 2D coords
     * @param {number} k - Number of neighbors to use
     * @returns {{x: number, y: number, cluster: number}} - Estimated position and cluster
     */
    estimatePosition(embedding, points, k = 5) {
        const neighbors = this.findNearestNeighbors(embedding, points, k);

        if (neighbors.length === 0) {
            return { x: 0, y: 0, cluster: 0 };
        }

        // Weighted average of neighbor positions
        let totalWeight = 0;
        let x = 0;
        let y = 0;
        const clusterCounts = {};

        for (const { point, similarity } of neighbors) {
            const viewData = point.views?.combined || point.views?.[Object.keys(point.views)[0]];
            if (!viewData) continue;

            // Use similarity as weight (squared for stronger influence of closer points)
            const weight = similarity * similarity;
            x += viewData.x * weight;
            y += viewData.y * weight;
            totalWeight += weight;

            // Count clusters for majority voting
            const cluster = viewData.cluster;
            clusterCounts[cluster] = (clusterCounts[cluster] || 0) + weight;
        }

        // Normalize (guard against division by zero)
        if (totalWeight > 0) {
            x /= totalWeight;
            y /= totalWeight;
        }

        // Find most likely cluster (weighted majority)
        let bestCluster = 0;
        let bestCount = 0;
        for (const [cluster, count] of Object.entries(clusterCounts)) {
            if (count > bestCount) {
                bestCount = count;
                bestCluster = parseInt(cluster);
            }
        }

        return { x, y, cluster: bestCluster };
    }
}

/**
 * Integration helper for main.js
 * Creates a new point object compatible with the visualization
 */
export function createNewPoint(id, nickname, responses, viewData) {
    return {
        id,
        nickname: nickname || 'new',
        responses,
        views: {
            combined: {
                x: viewData.x,
                y: viewData.y,
                cluster: viewData.cluster,
                text: Object.values(responses).join(' | '),
            },
        },
    };
}

/**
 * UI Helper: Create a simple input form for adding new points
 * @param {HTMLElement} container - Container element
 * @param {Function} onSubmit - Callback with form data
 */
export function createAddPointForm(container, onSubmit) {
    const form = document.createElement('div');
    form.className = 'add-point-form';
    form.innerHTML = `
        <h3>Add New Point</h3>
        <input type="text" id="new-nickname" placeholder="Nickname (optional)">
        <textarea id="new-response" placeholder="Enter your reflection..." rows="3"></textarea>
        <button id="add-point-btn">Add to Galaxy</button>
        <div id="add-point-status"></div>
    `;

    container.appendChild(form);

    const btn = form.querySelector('#add-point-btn');
    const input = form.querySelector('#new-response');
    const nicknameInput = form.querySelector('#new-nickname');
    const status = form.querySelector('#add-point-status');

    btn.addEventListener('click', async () => {
        const text = input.value.trim();
        if (!text) {
            status.textContent = 'Please enter some text';
            status.className = 'error';
            return;
        }

        status.textContent = 'Generating embedding...';
        status.className = 'loading';
        btn.disabled = true;

        try {
            await onSubmit({
                nickname: nicknameInput.value.trim() || 'anonymous',
                text,
            });

            status.textContent = 'Point added!';
            status.className = 'success';
            input.value = '';
            nicknameInput.value = '';

            // Clear status after 3 seconds
            setTimeout(() => {
                status.textContent = '';
            }, 3000);

        } catch (err) {
            status.textContent = `Error: ${err.message}`;
            status.className = 'error';
        } finally {
            btn.disabled = false;
        }
    });

    return form;
}
