/**
 * Ollama API Client for Gemini UI
 */

export class OllamaClient {
  constructor(baseUrl = 'http://localhost:11434') {
    this.baseUrl = baseUrl.replace(/\/$/, '');
  }

  setBaseUrl(url) {
    this.baseUrl = url.replace(/\/$/, '');
  }

  /**
   * Check if Ollama server is reachable via Proxy or Direct
   */
  async checkConnection() {
    // Try relative proxy first (Vite proxy)
    try {
      const proxyResp = await fetch('/api/tags', { method: 'GET' });
      if (proxyResp.ok) return true;
    } catch (e) {}

    // Fallback to direct fetch
    try {
      const targetUrl = this.baseUrl.startsWith('http') ? `${this.baseUrl}/api/tags` : '/api/tags';
      const response = await fetch(targetUrl, { method: 'GET' });
      return response.ok;
    } catch (err) {
      return false;
    }
  }

  /**
   * Fetch list of available models from Ollama
   */
  async getModels() {
    let response;
    
    // Try relative proxy first
    try {
      response = await fetch('/api/tags');
      if (!response.ok && this.baseUrl.startsWith('http')) {
        response = await fetch(`${this.baseUrl}/api/tags`);
      }
    } catch (e) {
      if (this.baseUrl.startsWith('http')) {
        response = await fetch(`${this.baseUrl}/api/tags`);
      } else {
        throw new Error('Connection refused. Is Ollama running?');
      }
    }

    if (!response || !response.ok) {
      throw new Error(`Failed to fetch models: HTTP ${response?.status || 'Error'}`);
    }

    const data = await response.json();
    return (data.models || []).map(m => m.name);
  }

  /**
   * Determine best working endpoint URL
   */
  async getEndpoint(path) {
    try {
      const test = await fetch('/api/tags', { method: 'GET' });
      if (test.ok) return path; // e.g. /api/chat
    } catch (e) {}

    return this.baseUrl.startsWith('http') ? `${this.baseUrl}${path}` : path;
  }

  /**
   * Stream chat completion from Ollama /api/chat
   */
  async streamChat({
    model,
    messages,
    systemPrompt = '',
    temperature = 0.7,
    numCtx = 4096,
    onChunk,
    onDone,
    onError,
    signal
  }) {
    try {
      // Construct full message history including system prompt if provided
      const chatPayload = [];
      if (systemPrompt && systemPrompt.trim()) {
        chatPayload.push({ role: 'system', content: systemPrompt.trim() });
      }
      
      messages.forEach(msg => {
        chatPayload.push({
          role: msg.role,
          content: msg.content
        });
      });

      const endpoint = await this.getEndpoint('/api/chat');

      const response = await fetch(endpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          model,
          messages: chatPayload,
          stream: true,
          options: {
            temperature: parseFloat(temperature),
            num_ctx: parseInt(numCtx, 10)
          }
        }),
        signal
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Ollama Chat Error (${response.status}): ${errorText || response.statusText}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder('utf-8');
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        
        // Process all complete lines
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (!line.trim()) continue;
          try {
            const parsed = JSON.parse(line);
            if (parsed.message && parsed.message.content) {
              onChunk(parsed.message.content);
            }
            if (parsed.done) {
              if (onDone) onDone();
              return;
            }
          } catch (jsonErr) {
            console.warn('NDJSON parse error:', jsonErr, line);
          }
        }
      }

      // Handle any remaining buffer
      if (buffer.trim()) {
        try {
          const parsed = JSON.parse(buffer);
          if (parsed.message && parsed.message.content) {
            onChunk(parsed.message.content);
          }
        } catch (e) {}
      }

      if (onDone) onDone();
    } catch (err) {
      if (err.name === 'AbortError') {
        console.log('Stream generation aborted by user.');
        if (onDone) onDone();
      } else {
        if (onError) onError(err);
      }
    }
  }
}
