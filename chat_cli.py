"""Interactive anonymous chat CLI with persistent sessions & conversations.

Features
--------
- Saves sessions (device id, client session id, cookies) to a local store, so
  the SAME anonymous session can be reused across runs instead of a new one
  every time.
- Saves conversations (conversation_id + parent message id + local history).
- Resume an old conversation, start a new one in the same session, or create a
  whole new session.
- Token is never written to disk; it is re-fetched on each run (cheap).

Usage:
  python3 chat_cli.py
Inside:
  /help for the command list
Store:
  ~/.chat_cli_store.json  (use --store to override)
"""

import json
import os
import sys
import time
import uuid
import gzip
import http.cookiejar
import urllib.error
import urllib.request
from pathlib import Path

BASE = 'https://android.chat.openai.com'
UA = 'ChatGPT/1.2026.293 (Android 12; sdk_gphone64_x86_64; build 2629319)'
DEFAULT_STORE = Path.home() / '.chat_cli_store.json'


# ---------------------------------------------------------------- store ----
def load_store(path):
    if path.exists():
        try:
            data = json.loads(path.read_text())
            data.setdefault('sessions', {})
            data.setdefault('conversations', {})
            data.setdefault('active_session', None)
            data.setdefault('active_conversation', None)
            return data
        except Exception:
            pass
    return {'sessions': {}, 'conversations': {},
            'active_session': None, 'active_conversation': None}


def save_store(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    os.chmod(tmp, 0o600)
    tmp.replace(path)


# -------------------------------------------------------------- session ----
class Session:
    def __init__(self, record=None):
        self.jar = http.cookiejar.CookieJar()
        self.opener = urllib.request.build_opener(
            urllib.request.HTTPCookieProcessor(self.jar))
        self.token = None
        self.conversation_id = None
        self.parent_id = str(uuid.uuid4())
        if record:
            self.id = record.get('id') or str(uuid.uuid4())[:8]
            self.device_id = record.get('device_id') or str(uuid.uuid4())
            self.client_session = record.get('client_session') or str(uuid.uuid4())
            self.created = record.get('created') or time.time()
            self.nick = record.get('nick') or ''
            for c in record.get('cookies', []):
                try:
                    self.jar.set_cookie(http.cookiejar.Cookie(
                        0, c['name'], c['value'], None, False,
                        c.get('domain', ''), True, False, c.get('path', '/'),
                        True, False, None, False, None, None, {}))
                except Exception:
                    pass
        else:
            self.id = str(uuid.uuid4())[:8]
            self.device_id = str(uuid.uuid4())
            self.client_session = str(uuid.uuid4())
            self.created = time.time()
            self.nick = ''

    def to_record(self):
        cookies = []
        for c in self.jar:
            cookies.append({'name': c.name, 'value': c.value,
                            'domain': c.domain, 'path': c.path})
        return {'id': self.id, 'device_id': self.device_id,
                'client_session': self.client_session,
                'created': self.created, 'nick': self.nick,
                'cookies': cookies}

    def headers(self, **extra):
        h = {
            'User-Agent': UA,
            'oai-package-name': 'com.openai.chatgpt',
            'oai-client-type': 'android',
            'oai-device-id': self.device_id,
            'oai-client-session': self.client_session,
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'identity',
        }
        h.update(extra)
        return h

    @staticmethod
    def _read(resp, limit=800000):
        raw = resp.read(limit)
        if resp.headers.get('Content-Encoding') == 'gzip':
            raw = gzip.decompress(raw)
        return raw

    def handshake(self):
        req = urllib.request.Request(
            BASE + '/backend-anon/sentinel/chat-requirements',
            data=b'{}', method='POST',
            headers=self.headers(Accept='application/json',
                                 **{'Content-Type': 'application/json'}))
        with self.opener.open(req, timeout=30) as r:
            data = json.loads(self._read(r))
        self.token = data.get('token')
        if not self.token:
            raise RuntimeError('لم يصدر الخادم توكن')
        return True

    def ask(self, text, conversation_id=None, parent_id=None):
        self.parent_id = parent_id or self.parent_id
        body = {
            'action': 'next',
            'messages': [{
                'id': str(uuid.uuid4()),
                'author': {'role': 'user'},
                'content': {'content_type': 'text', 'parts': [text]},
            }],
            'parent_message_id': self.parent_id,
            'model': 'auto',
            'timezone_offset_min': 180,
            'history_and_training_disabled': True,
            'conversation_mode': {'kind': 'primary_assistant'},
        }
        cid = conversation_id or self.conversation_id
        if cid:
            body['conversation_id'] = cid

        hdr = self.headers(
            Accept='text/event-stream',
            **{'Content-Type': 'application/json',
               'openai-sentinel-chat-requirements-token': self.token})

        req = urllib.request.Request(
            BASE + '/backend-anon/f/conversation',
            data=json.dumps(body).encode(), headers=hdr, method='POST')

        with self.opener.open(req, timeout=180) as r:
            raw = self._read(r)
        return self._parse_sse(raw.decode(errors='replace'))

    def _parse_sse(self, text):
        best, order, err = {}, [], None
        for line in text.splitlines():
            if not line.startswith('data:'):
                continue
            raw = line[5:].strip()
            if raw == '[DONE]':
                break
            try:
                ev = json.loads(raw)
            except ValueError:
                continue
            if not isinstance(ev, dict):
                continue
            if ev.get('error') and not err:
                err = str(ev['error'])[:300]
            if ev.get('conversation_id'):
                self.conversation_id = ev['conversation_id']
            v = ev.get('v')
            if isinstance(v, dict) and v.get('conversation_id'):
                self.conversation_id = v['conversation_id']
            m = ev.get('message')
            if not isinstance(m, dict):
                continue
            if m.get('author', {}).get('role') != 'assistant':
                continue
            mid = m.get('id') or 'anon'
            parts = m.get('content', {}).get('parts', [])
            joined = ''.join(p for p in parts if isinstance(p, str))
            if not joined.strip():
                continue
            if mid not in best:
                order.append(mid)
            if len(joined) >= len(best.get(mid, '')):
                best[mid] = joined
                self.parent_id = mid
        if not best:
            return ('[خطأ من الخادم] ' + err) if err else ''
        chunks = []
        for mid in order:
            t = best[mid]
            if t and t not in chunks:
                chunks.append(t)
        return ''.join(chunks)


# ------------------------------------------------------------ interface ----
def new_session(store, path, nick=''):
    s = Session()
    s.nick = nick
    print('… جارٍ إنشاء جلسة جديدة', end='', flush=True)
    s.handshake()
    print(' ✔ ' + s.id)
    store['sessions'][s.id] = s.to_record()
    store['active_session'] = s.id
    store['active_conversation'] = None
    save_store(path, store)
    return s


def use_session(store, path, sid):
    rec = store['sessions'].get(sid)
    if not rec:
        print('[خطأ] لا توجد جلسة بهذا المعرف')
        return None
    s = Session(rec)
    print('… جارٍ استئناف الجلسة ' + sid, end='', flush=True)
    s.handshake()
    print(' ✔')
    store['sessions'][s.id] = s.to_record()
    store['active_session'] = s.id
    save_store(path, store)
    return s


def new_conversation(store, path, sid, title=''):
    cid = 'c' + uuid.uuid4().hex[:7]
    store['conversations'][cid] = {
        'id': cid, 'session': sid, 'title': title or 'محادثة ' + cid,
        'conversation_id': None, 'parent_id': None,
        'created': time.time(), 'messages': []}
    store['active_conversation'] = cid
    save_store(path, store)
    return cid


def cmd_list(store):
    print('\n— الجلسات —')
    if not store['sessions']:
        print('  (لا يوجد)')
    for sid, r in store['sessions'].items():
        n = sum(1 for c in store['conversations'].values() if c['session'] == sid)
        mark = ' ←' if sid == store.get('active_session') else ''
        print('  ' + sid + '  ' + (r.get('nick') or '-') +
              '  محادثات:' + str(n) + mark)
    print('\n— المحادثات —')
    if not store['conversations']:
        print('  (لا يوجد)')
    for cid, c in store['conversations'].items():
        mark = ' ←' if cid == store.get('active_conversation') else ''
        print('  ' + cid + '  [' + c['session'] + ']  ' + c['title'] +
              '  (' + str(len(c['messages'])) + ' رسالة)' + mark)
    print()


def main():
    args = sys.argv[1:]
    path = DEFAULT_STORE
    if '--store' in args:
        i = args.index('--store')
        try:
            path = Path(args[i + 1]).expanduser()
            del args[i:i + 2]
        except IndexError:
            pass
    store = load_store(path)

    print()
    print('=== دردشة مجهولة — جلسات ومحادثات محفوظة ===')
    print('اكتب /help للأوامر')
    print()

    s = None
    if store.get('active_session') in store['sessions']:
        try:
            s = use_session(store, path, store['active_session'])
        except Exception as e:
            print('[تعذّر استئناف الجلسة] ' + str(e))
            s = None
    if not s:
        try:
            s = new_session(store, path)
        except Exception as e:
            print('[فشل] ' + type(e).__name__ + ': ' + str(e))
            sys.exit(1)

    conv = store.get('active_conversation')
    if conv not in store['conversations']:
        conv = new_conversation(store, path, s.id)
        print('✔ محادثة جديدة: ' + conv)
    else:
        print('↩ استئناف المحادثة: ' + conv +
              ' (' + store['conversations'][conv]['title'] + ')')
    print()

    while True:
        try:
            text = input('أنت > ').strip()
        except (EOFError, KeyboardInterrupt):
            print('\nمع السلامة.')
            break
        if not text:
            continue

        # ---- commands ----
        if text in ('/exit', '/quit', '/q'):
            print('مع السلامة.')
            break
        if text == '/help':
            print("""
/help                  هذه القائمة
/exit                  خروج

الجلسات (الهوية):
  /sessions            عرض الجلسات والمحادثات
  /newsession [اسم]    جلسة هوية جديدة تمامًا
  /session <id>        التبديل إلى جلسة محفوظة

المحادثات (السياق):
  /new [عنوان]         محادثة جديدة داخل الجلسة الحالية
  /open <id>           فتح محادثة محفوظة
  /title <نص>          تغيير عنوان المحادثة الحالية
  /history             عرض رسائل المحادثة الحالية
  /clear               مسح رسائل المحادثة محليًا
""")
            continue
        if text == '/sessions':
            cmd_list(store)
            continue
        if text.startswith('/newsession'):
            nick = text[len('/newsession'):].strip()
            try:
                s = new_session(store, path, nick)
                conv = new_conversation(store, path, s.id)
                print('✔ محادثة جديدة: ' + conv + '\n')
            except Exception as e:
                print('[فشل] ' + str(e))
            continue
        if text.startswith('/session '):
            sid = text.split(None, 1)[1].strip()
            try:
                ns = use_session(store, path, sid)
                if ns:
                    s = ns
                    mine = [c for c in store['conversations'] if
                            store['conversations'][c]['session'] == s.id]
                    if mine:
                        conv = mine[-1]
                        store['active_conversation'] = conv
                        save_store(path, store)
                        print('↩ المحادثة: ' + conv + '\n')
                    else:
                        conv = new_conversation(store, path, s.id)
                        print('✔ محادثة جديدة: ' + conv + '\n')
            except Exception as e:
                print('[فشل] ' + str(e))
            continue
        if text.startswith('/new') and not text.startswith('/newsession'):
            title = text[len('/new'):].strip()
            conv = new_conversation(store, path, s.id, title)
            print('✔ محادثة جديدة: ' + conv +
                  (' — ' + title if title else '') + '\n')
            continue
        if text.startswith('/open '):
            cid = text.split(None, 1)[1].strip()
            if cid not in store['conversations']:
                print('[خطأ] لا توجد محادثة بهذا المعرف\n')
                continue
            c = store['conversations'][cid]
            if c['session'] != s.id:
                try:
                    ns = use_session(store, path, c['session'])
                    if not ns:
                        continue
                    s = ns
                except Exception as e:
                    print('[فشل] ' + str(e))
                    continue
            conv = cid
            store['active_conversation'] = conv
            save_store(path, store)
            print('↩ فتحت: ' + c['title'] +
                  ' (' + str(len(c['messages'])) + ' رسالة)\n')
            continue
        if text.startswith('/title '):
            t = text.split(None, 1)[1].strip()
            store['conversations'][conv]['title'] = t
            save_store(path, store)
            print('✔ العنوان: ' + t + '\n')
            continue
        if text == '/history':
            msgs = store['conversations'][conv]['messages']
            if not msgs:
                print('(المحادثة فارغة)\n')
            for m in msgs:
                who = 'أنت' if m['role'] == 'user' else 'ChatGPT'
                print(who + ' > ' + m['text'])
                print()
            continue
        if text == '/clear':
            store['conversations'][conv]['messages'] = []
            save_store(path, store)
            print('✔ مُسحت الرسائل محليًا (السياق على الخادم كما هو)\n')
            continue
        if text.startswith('/'):
            print('أمر غير معروف — /help\n')
            continue

        # ---- send ----
        c = store['conversations'][conv]
        print('…', end='', flush=True)
        try:
            reply = s.ask(text, c.get('conversation_id'), c.get('parent_id'))
        except urllib.error.HTTPError as e:
            print()
            print('[خطأ HTTP ' + str(e.code) + '] ' +
                  e.read(400).decode(errors='replace')[:300])
            print()
            continue
        except Exception as e:
            print()
            print('[خطأ] ' + type(e).__name__ + ': ' + str(e))
            print()
            continue

        print('\r   \r', end='')
        print('ChatGPT > ' + (reply if reply else '(رد فارغ)'))
        print()

        c = store['conversations'][conv]
        c['messages'].append({'role': 'user', 'text': text,
                              't': time.time()})
        c['messages'].append({'role': 'assistant', 'text': reply,
                              't': time.time()})
        if s.conversation_id:
            c['conversation_id'] = s.conversation_id
        c['parent_id'] = s.parent_id
        store['sessions'][s.id] = s.to_record()
        save_store(path, store)


if __name__ == '__main__':
    main()
