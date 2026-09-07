import { defineConfig } from 'vite';
import https from 'https';

export default defineConfig({
  server: {
    host: '0.0.0.0',
    port: 3000,
    proxy: {
      '/api/ingest': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
        secure: false,
      },
      '/api/rag': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
        secure: false,
      },
      '/api': {
        target: 'http://127.0.0.1:11434',
        changeOrigin: true,
        secure: false,
        bypass(req) {
          if (req.url && (req.url.startsWith('/api/search') || req.url.startsWith('/api/rag') || req.url.startsWith('/api/ingest'))) return req.url;
          return null;
        },
        configure: (proxy) => {
          proxy.on('proxyReq', (proxyReq) => {
            proxyReq.setHeader('origin', 'http://127.0.0.1:11434');
          });
        }
      }
    }
  },
  plugins: [
    {
      name: 'web-search-plugin',
      configureServer(server) {
        server.middlewares.use('/api/search', (req, res) => {
          const url = new URL(req.url, `http://${req.headers.host}`);
          const q = url.searchParams.get('q');

          if (!q) {
            res.statusCode = 400;
            res.setHeader('Content-Type', 'application/json');
            return res.end(JSON.stringify({ error: 'Missing query' }));
          }

          const postData = `q=${encodeURIComponent(q)}`;
          const options = {
            hostname: 'lite.duckduckgo.com',
            path: '/lite/',
            method: 'POST',
            headers: {
              'Content-Type': 'application/x-www-form-urlencoded',
              'Content-Length': Buffer.byteLength(postData),
              'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
          };

          const ddgReq = https.request(options, (ddgRes) => {
            let body = '';
            ddgRes.on('data', chunk => body += chunk);
            ddgRes.on('end', () => {
              const results = [];
              const parts = body.split('class="result-snippet"');
              for (let i = 1; i < parts.length && i <= 3; i++) {
                let text = parts[i].split('>')[1]?.split('<')[0];
                if (text && text.trim()) results.push(text.trim());
              }
              res.setHeader('Content-Type', 'application/json');
              res.end(JSON.stringify({ results }));
            });
          });

          ddgReq.on('error', (e) => {
            res.statusCode = 500;
            res.setHeader('Content-Type', 'application/json');
            res.end(JSON.stringify({ error: e.message }));
          });

          ddgReq.write(postData);
          ddgReq.end();
        });
      }
    }
  ]
});
