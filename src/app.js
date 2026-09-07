import { OllamaClient } from './ollama.js';

class App {
  constructor() {
    this.settings = this.loadSettings();
    this.client   = new OllamaClient(this.settings.host);
    this.chats    = this.loadChats();
    this.activeId = null;
    this.generating = false;
    this.abort    = null;
    this.models   = [];
    this.webSearchEnabled = false;
    this.ragEnabled = false;

    this.$ = id => document.getElementById(id);
    this.q = sel => document.querySelector(sel);

    this.initMarked();
    this.render();
    this.bindAll();
    this.boot();
  }

  /* ── Persistence ──────────────────── */
  loadSettings() {
    try {
      return JSON.parse(localStorage.getItem('oc_settings')) || {};
    } catch { return {}; }
  }
  get cfg() {
    return {
      host:   this.settings.host   || 'http://localhost:11434',
      sys:    this.settings.sys    || 'คุณคือผู้ช่วย AI ที่ฉลาด เป็นกันเอง และตอบเป็นภาษาไทยเสมอ ห้ามใช้ตัวอักษรจีนหรือภาษาจีนเด็ดขาด หากมีคำศัพท์ทับศัพท์ให้ใช้ภาษาไทยหรือภาษาอังกฤษเท่านั้น',
      temp:   this.settings.temp   ?? 0.7,
      ctx:    this.settings.ctx    || 4096,
    };
  }
  saveSettings() { localStorage.setItem('oc_settings', JSON.stringify(this.settings)); }

  loadChats() {
    try { return JSON.parse(localStorage.getItem('oc_chats')) || []; } catch { return []; }
  }
  saveChats() { localStorage.setItem('oc_chats', JSON.stringify(this.chats)); }

  /* ── Markdown ─────────────────────── */
  initMarked() {
    if (!window.marked || !window.hljs) return;
    const renderer = new window.marked.Renderer();
    renderer.code = (code, lang) => {
      const l = window.hljs.getLanguage(lang) ? lang : 'plaintext';
      const highlighted = window.hljs.highlight(code, { language: l }).value;
      return `<div class="code-block">
        <div class="code-header">
          <span>${l}</span>
          <button class="copy-btn" onclick="(()=>{navigator.clipboard.writeText(decodeURIComponent('${encodeURIComponent(code)}'));this.textContent='✓ คัดลอกแล้ว';setTimeout(()=>this.textContent='คัดลอก',2000)}).call(this)">คัดลอก</button>
        </div>
        <pre><code class="hljs language-${l}">${highlighted}</code></pre>
      </div>`;
    };
    window.marked.setOptions({ renderer });
  }

  md(text) {
    if (!text) return '';
    // Handle <think> tags
    let out = text.replace(/<think>([\s\S]*?)<\/think>/g, (_, c) =>
      `<div class="think-block">
        <button class="think-toggle"><span class="think-dot"></span>กระบวนการคิด (คลิกเพื่อดู)</button>
        <div class="think-body">${window.marked ? window.marked.parse(c.trim()) : c}</div>
      </div>`
    ).replace(/<think>([\s\S]*)$/, (_, c) =>
      `<div class="think-block">
        <button class="think-toggle"><span class="think-dot"></span>กำลังคิด...</button>
        <div class="think-body">${window.marked ? window.marked.parse(c.trim()) : c}</div>
      </div>`
    );
    return window.marked ? window.marked.parse(out) : out.replace(/</g,'&lt;');
  }

  /* ── Sidebar ──────────────────────── */
  renderSidebar() {
    const list = this.$('chat-list');
    if (!list) return;
    list.innerHTML = '';
    this.chats.forEach(c => {
      const el = document.createElement('div');
      el.className = 'chat-item' + (c.id === this.activeId ? ' active' : '');
      el.innerHTML = `
        <span class="chat-item-title">${this.esc(c.title)}</span>
        <button class="chat-item-del" data-id="${c.id}" title="ลบ">
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6"/>
            <path d="M10 11v6M14 11v6"/>
          </svg>
        </button>`;
      el.addEventListener('click', e => {
        if (e.target.closest('.chat-item-del')) {
          this.deleteChat(c.id);
        } else {
          this.selectChat(c.id);
        }
      });
      list.appendChild(el);
    });
  }

  /* ── Boot / Init ──────────────────── */
  async boot() {
    await this.checkConn();
    await this.fetchModels();
    this.renderSidebar();
    if (this.chats.length) this.selectChat(this.chats[0].id);
    else this.newChat();
  }

  /* ── Connection ───────────────────── */
  async checkConn() {
    const ok = await this.client.checkConnection();
    const dot  = this.$('conn-dot');
    const text = this.$('conn-text');
    if (dot)  dot.className  = 'conn-dot ' + (ok ? 'online' : 'offline');
    if (text) text.textContent = ok ? 'Ollama: ออนไลน์' : 'Ollama: ออฟไลน์';
    return ok;
  }

  /* ── Models ───────────────────────── */
  async fetchModels() {
    const sel = this.$('model-select');
    if (!sel) return;
    sel.innerHTML = '<option value="" disabled selected>กำลังโหลด...</option>';
    try {
      this.models = await this.client.getModels();
      if (!this.models.length) {
        sel.innerHTML = '<option value="" disabled selected>ไม่พบโมเดล — กด "ดึงโมเดล"</option>';
        return;
      }
      sel.innerHTML = this.models.map(m => `<option value="${m}">${m}</option>`).join('');
      const activeChat = this.chats.find(c => c.id === this.activeId);
      const preferred  = this.models.find(m => m.startsWith('qwen2.5'));
      if (activeChat?.model && this.models.includes(activeChat.model)) {
        sel.value = activeChat.model;
      } else if (preferred) {
        sel.value = preferred;
      } else {
        sel.value = this.models[0];
      }
    } catch {
      sel.innerHTML = '<option value="" disabled selected>โหลดล้มเหลว</option>';
    }
  }

  currentModel() { return this.$('model-select')?.value || ''; }

  /* ── Chat Management ──────────────── */
  newChat() {
    const c = {
      id: 'c' + Date.now(),
      title: 'แชทใหม่',
      model: this.currentModel() || (this.models[0] || ''),
      messages: [],
    };
    this.chats.unshift(c);
    this.saveChats();
    this.renderSidebar();
    this.selectChat(c.id);
  }

  selectChat(id) {
    this.activeId = id;
    const chat = this.chats.find(c => c.id === id);
    if (!chat) return;
    if (chat.model && this.models.includes(chat.model)) {
      const sel = this.$('model-select');
      if (sel) sel.value = chat.model;
    }
    this.renderSidebar();
    this.renderMessages(chat);
  }

  deleteChat(id) {
    this.chats = this.chats.filter(c => c.id !== id);
    this.saveChats();
    if (this.activeId === id) {
      if (this.chats.length) this.selectChat(this.chats[0].id);
      else this.newChat();
    } else {
      this.renderSidebar();
    }
  }

  /* ── Message Rendering ────────────── */
  renderMessages(chat) {
    const welcome  = this.$('welcome');
    const messages = this.$('messages');
    if (!welcome || !messages) return;

    messages.innerHTML = '';

    if (!chat || chat.messages.length === 0) {
      welcome.classList.remove('hidden');
      return;
    }
    welcome.classList.add('hidden');

    chat.messages.forEach(m => this.appendMsg(m.role, m.content, false, m.sources || []));
    this.scrollBottom();
  }

  appendMsg(role, content, streaming = false, sources = []) {
    const welcome = this.$('welcome');
    if (welcome) welcome.classList.add('hidden');

    const messages = this.$('messages');
    if (!messages) return null;

    const row = document.createElement('div');
    row.className = `msg ${role}`;

    if (role === 'user') {
      row.innerHTML = `
        <div class="msg-body"><div class="msg-bubble">${this.esc(content)}</div></div>
        <div class="msg-avatar">U</div>`;
    } else {
      row.innerHTML = `
        <div class="msg-avatar">AI</div>
        <div class="msg-body">
          <div class="msg-content ${streaming ? 'cursor' : ''}">${this.md(content)}</div>
          ${this.renderRagSources(sources, content)}
        </div>`;
    }

    messages.appendChild(row);
    this.bindThinkToggles(row);
    this.scrollBottom();
    return row.querySelector(role === 'user' ? '.msg-bubble' : '.msg-content');
  }

  renderRagSources(sources, answer = '') {
    if (!sources?.length) return '';
    const cited = new Set((answer.match(/\[S\d+\]/g) || []).map(s => s.slice(1, -1)));
    const known = new Set(sources.map(s => s.citation_id));
    const unknown = [...cited].filter(id => !known.has(id));
    const rows = sources.map(source => {
      const status = cited.has(source.citation_id) ? 'อ้างอิงในคำตอบ' : 'ค้นพบจากการค้นคืน';
      const score = source.rerank_score == null ? '' : ` · rerank ${Number(source.rerank_score).toFixed(3)}`;
      return `<li><code>[${this.esc(source.citation_id)}]</code> ${this.esc(source.source_name || source.source_id || 'ไม่ทราบแหล่งข้อมูล')} · ${this.esc(source.chunk_id || source.document_id || '-')}${score} <span>${status}</span></li>`;
    }).join('');
    const warning = cited.size === 0
      ? '<p class="rag-source-warning">ไม่พบ citation ในคำตอบ — โปรดตรวจสอบการรองรับของข้อเท็จจริง</p>'
      : (unknown.length ? `<p class="rag-source-warning">พบ citation ที่ไม่ตรงกับหลักฐาน: ${this.esc(unknown.join(', '))}</p>` : '');
    return `<div class="rag-sources"><div class="rag-sources-title">แหล่งข้อมูลที่ค้นพบ</div><ul>${rows}</ul>${warning}</div>`;
  }

  bindThinkToggles(el) {
    el.querySelectorAll('.think-toggle').forEach(btn => {
      btn.addEventListener('click', () => btn.parentElement.classList.toggle('collapsed'));
    });
  }

  scrollBottom() {
    const area = this.$('chat-area');
    if (area) area.scrollTop = area.scrollHeight;
  }

  esc(s) {
    return String(s)
      .replace(/&/g,'&amp;').replace(/</g,'&lt;')
      .replace(/>/g,'&gt;').replace(/"/g,'&quot;');
  }

  /* ── Send ─────────────────────────── */
  async send() {
    if (this.generating) {
      this.abort?.abort();
      return;
    }

    const inp   = this.$('user-input');
    const text  = inp?.value.trim();
    const model = this.currentModel();

    if (!text) return;
    if (!model) { alert('กรุณาเลือกหรือดึงโมเดลก่อนครับ'); return; }

    let chat = this.chats.find(c => c.id === this.activeId);
    if (!chat) { this.newChat(); chat = this.chats[0]; }

    if (chat.messages.length === 0) {
      chat.title = text.length > 35 ? text.slice(0, 35) + '...' : text;
      this.renderSidebar();
    }

    chat.messages.push({ role: 'user', content: text });
    this.saveChats();

    inp.value = '';
    inp.style.height = 'auto';
    this.appendMsg('user', text, false);

    // AI placeholder
    const aiBody = this.appendMsg('assistant', '', true);

    this.setGenerating(true);
    this.abort = new AbortController();

    let full = '';
    let ragSources = [];
    
    // Multimodal RAG Logic (ChromaDB)
    let ragContext = '';
    if (this.ragEnabled) {
      if (aiBody) aiBody.innerHTML = this.md('🧠กำลังค้นหาบริบทความรู้จาก ChromaDB (ViT Vector Store)...');
      try {
        const rRes = await fetch('/api/rag/query', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            prompt: text,
            n_results: 3,
            // History is sent only for intent/coreference rewriting. The
            // backend excludes assistant turns from factual query context.
            conversation_context: chat.messages.slice(0, -1).slice(-8)
          })
        });
        if (rRes.ok) {
          const rData = await rRes.json();
          if (rData.context_str) {
            ragContext = rData.context_str;
          }
          ragSources = Array.isArray(rData.sources) ? rData.sources : [];
        }
      } catch (e) {
        console.warn('RAG search failed:', e);
      }
      if (aiBody) aiBody.innerHTML = '';
    }

    // Web Search Logic
    let searchContext = '';
    if (this.webSearchEnabled) {
      if (aiBody) aiBody.innerHTML = this.md('🔍กำลังค้นหาข้อมูลบนอินเทอร์เน็ต...');
      try {
        const sRes = await fetch(`/api/search?q=${encodeURIComponent(text)}`);
        if (sRes.ok) {
          const sData = await sRes.json();
          if (sData.results && sData.results.length > 0) {
            searchContext = `\n\n[ข้อมูลเสริมจากการค้นหาอินเทอร์เน็ตล่าสุด:\n- ${sData.results.join('\n- ')}\n]\n\nโปรดใช้ข้อมูลข้างต้นในการตอบคำถามหากเกี่ยวข้อง`;
          }
        }
      } catch (e) {
        console.warn('Web search failed:', e);
      }
      if (aiBody) aiBody.innerHTML = '';
    }

    const messagesForApi = chat.messages.map((m, i) => {
      if (i === chat.messages.length - 1) {
        let appended = m.content;
        if (ragContext) appended += ragContext;
        if (searchContext) appended += searchContext;
        return { ...m, content: appended };
      }
      return m;
    });

    await this.client.streamChat({
      model,
      messages: messagesForApi,
      systemPrompt: this.cfg.sys,
      temperature:  this.cfg.temp,
      numCtx:       this.cfg.ctx,
      signal:       this.abort.signal,
      onChunk: chunk => {
        full += chunk;
        if (aiBody) {
          aiBody.innerHTML = this.md(full);
          aiBody.classList.add('cursor');
          this.bindThinkToggles(aiBody.parentElement);
          this.scrollBottom();
        }
      },
      onDone: () => {
        this.setGenerating(false);
        if (aiBody) aiBody.classList.remove('cursor');
        if (full.trim()) {
          chat.messages.push({ role: 'assistant', content: full, sources: ragSources });
          if (aiBody) {
            aiBody.innerHTML = this.md(full);
            const body = aiBody.parentElement;
            body?.querySelector('.rag-sources')?.remove();
            if (ragSources.length) body?.insertAdjacentHTML('beforeend', this.renderRagSources(ragSources, full));
          }
          this.saveChats();
        }
      },
      onError: err => {
        this.setGenerating(false);
        if (aiBody) {
          aiBody.classList.remove('cursor');
          const errMsg = `> ⚠️ **ข้อผิดพลาด**: ${err.message}`;
          full += '\n\n' + errMsg;
          aiBody.innerHTML = this.md(full);
        }
        chat.messages.push({ role: 'assistant', content: full });
        this.saveChats();
      }
    });
  }

  setGenerating(v) {
    this.generating = v;
    const btn  = this.$('send-btn');
    const si   = btn?.querySelector('.send-icon');
    const stop = btn?.querySelector('.stop-icon');
    if (v) {
      btn?.classList.add('stop');
      si?.classList.add('hidden');
      stop?.classList.remove('hidden');
    } else {
      btn?.classList.remove('stop');
      si?.classList.remove('hidden');
      stop?.classList.add('hidden');
    }
  }

  /* ── Settings ─────────────────────── */
  openSettings() {
    const m = this.$('settings-modal');
    if (!m) return;
    this.$('host-input').value     = this.cfg.host;
    this.$('sysprompt-input').value = this.cfg.sys;
    this.$('temp-input').value     = this.cfg.temp;
    this.$('temp-label').textContent = this.cfg.temp;
    this.$('ctx-input').value      = this.cfg.ctx;
    m.classList.remove('hidden');
  }
  closeSettings() { this.$('settings-modal')?.classList.add('hidden'); }

  async saveSettings() {
    this.settings = {
      host: this.$('host-input').value.trim() || 'http://localhost:11434',
      sys:  this.$('sysprompt-input').value.trim(),
      temp: parseFloat(this.$('temp-input').value),
      ctx:  parseInt(this.$('ctx-input').value),
    };
    this.saveSettings_persist();
    this.client.setBaseUrl(this.settings.host);
    this.closeSettings();
    await this.checkConn();
    await this.fetchModels();
  }
  saveSettings_persist() { localStorage.setItem('oc_settings', JSON.stringify(this.settings)); }

  /* ── Pull Modal ───────────────────── */
  openPull() {
    const m = this.$('pull-modal');
    if (!m) return;
    this.$('pull-name-input').value = 'qwen2.5:7b';
    this.$('pull-progress')?.classList.add('hidden');
    if (this.$('progress-fill')) this.$('progress-fill').style.width = '0%';
    if (this.$('progress-fill')) this.$('progress-fill').classList.remove('indeterminate');
    if (this.$('start-pull-btn')) this.$('start-pull-btn').disabled = false;
    document.querySelectorAll('.chip').forEach(c => c.classList.remove('selected'));
    const def = document.querySelector('.chip[data-model="qwen2.5:7b"]');
    if (def) def.classList.add('selected');
    m.classList.remove('hidden');
  }
  closePull() { this.$('pull-modal')?.classList.add('hidden'); }

  async startPull() {
    const name = this.$('pull-name-input')?.value.trim();
    if (!name) { this.$('pull-name-input')?.focus(); return; }

    const startBtn   = this.$('start-pull-btn');
    const progress   = this.$('pull-progress');
    const statusEl   = this.$('progress-status');
    const fillEl     = this.$('progress-fill');
    const pctEl      = this.$('progress-pct');

    if (startBtn)  startBtn.disabled = true;
    if (progress)  progress.classList.remove('hidden');
    if (statusEl)  statusEl.textContent = `กำลังดาวน์โหลด ${name}...`;
    if (fillEl)    { fillEl.classList.add('indeterminate'); fillEl.style.width = '55%'; }
    if (pctEl)     pctEl.textContent = 'กำลังเริ่มต้น...';

    try {
      const endpoint = await this.client.getEndpoint('/api/pull');
      const res = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name, stream: true }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);

      const reader  = res.body.getReader();
      const decoder = new TextDecoder();
      let buf = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        const lines = buf.split('\n'); buf = lines.pop() || '';
        for (const line of lines) {
          if (!line.trim()) continue;
          try {
            const d = JSON.parse(line);
            if (d.status && statusEl) statusEl.textContent = d.status;
            if (d.total && d.completed) {
              const pct = Math.round(d.completed / d.total * 100);
              if (fillEl) { fillEl.classList.remove('indeterminate'); fillEl.style.width = pct + '%'; }
              if (pctEl)  pctEl.textContent = pct + '%';
            }
          } catch {}
        }
      }

      if (fillEl) { fillEl.classList.remove('indeterminate'); fillEl.style.width = '100%'; }
      if (pctEl)  pctEl.textContent = '100%';
      if (statusEl) statusEl.textContent = `✅ ดาวน์โหลด ${name} สำเร็จ!`;

      await this.fetchModels();
      await this.checkConn();
      setTimeout(() => this.closePull(), 1800);

    } catch (err) {
      if (fillEl) fillEl.classList.remove('indeterminate');
      if (statusEl) statusEl.textContent = `❌ ล้มเหลว: ${err.message}`;
      if (pctEl)  pctEl.textContent = 'ข้อผิดพลาด';
      if (startBtn) startBtn.disabled = false;
    }
  }

  /* ── Render & Bind ────────────────── */
  render() {
    // Initial render before boot
  }

  bindAll() {
    // Sidebar
    this.$('new-chat-btn')?.addEventListener('click', () => this.newChat());
    this.$('toggle-sidebar-btn')?.addEventListener('click', () => {
      document.querySelector('.sidebar')?.classList.toggle('hidden');
    });
    this.$('refresh-conn-btn')?.addEventListener('click', () => {
      this.checkConn();
      this.fetchModels();
    });
    this.$('settings-btn')?.addEventListener('click', () => this.openSettings());

    // Model picker
    this.$('refresh-models-btn')?.addEventListener('click', () => this.fetchModels());
    this.$('model-select')?.addEventListener('change', e => {
      const chat = this.chats.find(c => c.id === this.activeId);
      if (chat) { chat.model = e.target.value; this.saveChats(); }
    });

    // Pull
    this.$('pull-btn')?.addEventListener('click', () => this.openPull());
    this.$('close-pull-btn')?.addEventListener('click', () => this.closePull());
    this.$('cancel-pull-btn')?.addEventListener('click', () => this.closePull());
    this.$('start-pull-btn')?.addEventListener('click', () => this.startPull());

    document.querySelectorAll('.chip').forEach(chip => {
      chip.addEventListener('click', () => {
        document.querySelectorAll('.chip').forEach(c => c.classList.remove('selected'));
        chip.classList.add('selected');
        const inp = this.$('pull-name-input');
        if (inp) inp.value = chip.getAttribute('data-model');
      });
    });

    // Web Search Toggle
    this.$('web-search-toggle')?.addEventListener('click', e => {
      this.webSearchEnabled = !this.webSearchEnabled;
      const btn = e.currentTarget;
      if (this.webSearchEnabled) btn.classList.add('active');
      else btn.classList.remove('active');
    });

    // RAG Toggle
    this.$('rag-toggle')?.addEventListener('click', e => {
      this.ragEnabled = !this.ragEnabled;
      const btn = e.currentTarget;
      if (this.ragEnabled) btn.classList.add('active');
      else btn.classList.remove('active');
    });

    // KB Modal Controls
    const kbModal = this.$('kb-modal');
    const loadKbStats = async () => {
      try {
        const res = await fetch('/api/rag/stats');
        if (res.ok) {
          const stats = await res.json();
          if (this.$('kb-text-count')) this.$('kb-text-count').textContent = stats.text_count || 0;
          if (this.$('kb-img-count')) this.$('kb-img-count').textContent = stats.image_count || 0;
        }
      } catch {}
    };

    this.$('kb-btn')?.addEventListener('click', () => {
      kbModal?.classList.remove('hidden');
      loadKbStats();
    });

    this.$('close-kb-modal')?.addEventListener('click', () => {
      kbModal?.classList.add('hidden');
    });

    // KB Tab switching
    this.$('kb-tab-text')?.addEventListener('click', () => {
      this.$('kb-tab-text')?.classList.add('active');
      this.$('kb-tab-image')?.classList.remove('active');
      this.$('kb-sec-text')?.classList.remove('hidden');
      this.$('kb-sec-image')?.classList.add('hidden');
    });

    this.$('kb-tab-image')?.addEventListener('click', () => {
      this.$('kb-tab-image')?.classList.add('active');
      this.$('kb-tab-text')?.classList.remove('active');
      this.$('kb-sec-image')?.classList.remove('hidden');
      this.$('kb-sec-text')?.classList.add('hidden');
    });

    // Ingest Text
    this.$('kb-save-text-btn')?.addEventListener('click', async () => {
      const txtInput = this.$('kb-text-input');
      const statusEl = this.$('kb-status-msg');
      const text = txtInput?.value.trim();
      if (!text) return;

      if (statusEl) statusEl.textContent = '⏳ กำลังสร้าง Vector Embedding และบันทึกลง ChromaDB...';
      try {
        const res = await fetch('/api/ingest/text', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text, source_name: 'user_kb_input' })
        });
        if (res.ok) {
          const d = await res.json();
          if (statusEl) statusEl.textContent = `✅ บันทึกสำเร็จ! เพิ่ม ${d.chunks_added} chunks เข้า ChromaDB`;
          if (txtInput) txtInput.value = '';
          loadKbStats();
        } else {
          if (statusEl) statusEl.textContent = '❌ บันทึกล้มเหลว';
        }
      } catch (e) {
        if (statusEl) statusEl.textContent = `❌ เกิดข้อผิดพลาด: ${e.message}`;
      }
    });

    // Ingest Image
    this.$('kb-save-image-btn')?.addEventListener('click', async () => {
      const fileInput = this.$('kb-image-input');
      const statusEl = this.$('kb-status-msg');
      const file = fileInput?.files?.[0];
      if (!file) {
        if (statusEl) statusEl.textContent = 'กรุณาเลือกไฟล์รูปภาพก่อนครับ';
        return;
      }

      if (statusEl) statusEl.textContent = '⏳ กำลังประมวลผลด้วย ViT / CLIP และบันทึกลง ChromaDB...';
      try {
        const formData = new FormData();
        formData.append('file', file);

        const res = await fetch('/api/ingest/image', {
          method: 'POST',
          body: formData
        });
        if (res.ok) {
          const d = await res.json();
          if (statusEl) statusEl.textContent = `✅ บันทึกสำเร็จ! เพิ่มรูป ${d.image_name} เข้า ChromaDB`;
          if (fileInput) fileInput.value = '';
          loadKbStats();
        } else {
          if (statusEl) statusEl.textContent = '❌ บันทึกล้มเหลว';
        }
      } catch (e) {
        if (statusEl) statusEl.textContent = `❌ เกิดข้อผิดพลาด: ${e.message}`;
      }
    });

    // Send
    this.$('send-btn')?.addEventListener('click', () => this.send());
    this.$('user-input')?.addEventListener('keydown', e => {
      if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); this.send(); }
    });
    this.$('user-input')?.addEventListener('input', e => {
      e.target.style.height = 'auto';
      e.target.style.height = Math.min(e.target.scrollHeight, 180) + 'px';
    });

    // Settings
    this.$('close-settings-btn')?.addEventListener('click', () => this.closeSettings());
    this.$('cancel-settings-btn')?.addEventListener('click', () => this.closeSettings());
    this.$('save-settings-btn')?.addEventListener('click', () => this.saveSettings());
    this.$('temp-input')?.addEventListener('input', e => {
      const lbl = this.$('temp-label');
      if (lbl) lbl.textContent = e.target.value;
    });

    // Suggestion cards
    document.querySelectorAll('.suggestion').forEach(btn => {
      btn.addEventListener('click', () => {
        const inp = this.$('user-input');
        if (inp) { inp.value = btn.getAttribute('data-prompt') || ''; this.send(); }
      });
    });

    // Close modals on backdrop click
    this.$('settings-modal')?.addEventListener('click', e => {
      if (e.target === this.$('settings-modal')) this.closeSettings();
    });
    this.$('pull-modal')?.addEventListener('click', e => {
      if (e.target === this.$('pull-modal')) this.closePull();
    });
  }
}

document.addEventListener('DOMContentLoaded', () => { window.app = new App(); });
