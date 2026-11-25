#!/usr/bin/env bash
# llama.cpp-inkling pipeline (macOS, Apple Silicon, Metal).
#
#   inkling/inkling.sh sync [tag]       import an Unsloth llama.cpp release into branch `unsloth`, merge it into this branch
#   inkling/inkling.sh build            static Metal build of this checkout -> build-inkling/bin (stable path for apps)
#   inkling/inkling.sh check            op tests; with MODEL=<gguf>: speed and KLD against the stock build of the same release
#   inkling/inkling.sh install studio   build, then the Unsloth Studio steps
#   inkling/inkling.sh install server   build, then run llama-server as a login service (OpenAI API for any app);
#                                       HOST=0.0.0.0 opens it to your network, always behind an API key
#   inkling/inkling.sh uninstall server
set -euo pipefail

ROOT=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
cd "$ROOT"
OUT=$ROOT/build-inkling
BIN=$OUT/bin
REPO=unslothai/llama.cpp
LABEL=llama.cpp-inkling.server
PLIST=$HOME/Library/LaunchAgents/$LABEL.plist
KEYFILE=$HOME/.config/llama.cpp-inkling/api-key

die() { echo "error: $*" >&2; exit 1; }

cmake_build() { # <src> <build dir> <targets...>
  local src=$1 dir=$2; shift 2
  local fix=() # some Command Line Tools installs lack the libc++ headers: use the SDK's copy
  echo '#include <array>' | c++ -x c++ -fsyntax-only - 2>/dev/null ||
    fix=(-DCMAKE_CXX_FLAGS="-isystem $(xcrun --show-sdk-path)/usr/include/c++/v1")
  cmake -S "$src" -B "$dir" -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=ON ${fix[@]+"${fix[@]}"} >/dev/null
  cmake --build "$dir" -j "$(sysctl -n hw.ncpu)" --target "$@" >/dev/null
}

cmd_sync() {
  local tag=${1:-}
  [ -n "$tag" ] || tag=$(curl -fsSL "https://api.github.com/repos/$REPO/releases/latest" | sed -n 's/.*"tag_name": *"\([^"]*\)".*/\1/p')
  [ -n "$tag" ] || die "no release tag"
  local base=${tag%%-mix-*}
  if git rev-parse -q --verify HEAD >/dev/null; then git diff --quiet HEAD || die "commit or stash your changes first"; fi
  if git rev-parse -q --verify "refs/tags/unsloth-$tag" >/dev/null; then
    echo "==> $tag already imported"
  else
    echo "==> import Unsloth $tag (ggml-org $base + Unsloth's PR set)"
    git remote get-url ggml >/dev/null 2>&1 || git remote add ggml https://github.com/ggml-org/llama.cpp.git
    git fetch -q --no-tags --filter=blob:none ggml "refs/tags/$base:refs/tags/ggml-$base"
    local tmp; tmp=$(mktemp -d)
    curl -fsSL "https://github.com/$REPO/releases/download/$tag/llama.cpp-source-$tag.tar.gz" | tar -xz -C "$tmp" --strip-components=1
    local tree
    tree=$(export GIT_INDEX_FILE=$tmp.index; git -c core.excludesFile=/dev/null --work-tree="$tmp" add -A && git write-tree)
    rm -rf "$tmp" "$tmp.index"
    local parents=()
    git rev-parse -q --verify refs/heads/unsloth >/dev/null && parents+=(-p unsloth)
    parents+=(-p "ggml-$base")
    local c; c=$(git commit-tree "$tree" "${parents[@]}" -m "unsloth $tag" -m "Source of https://github.com/$REPO/releases/tag/$tag (ggml-org $base + Unsloth's PR set).")
    git update-ref refs/heads/unsloth "$c"
    git tag "unsloth-$tag" "$c"
  fi
  [ "$(git branch --show-current)" = unsloth ] && return
  echo "==> merge into $(git branch --show-current)"
  git merge --no-edit -m "Merge Unsloth $tag" "unsloth-$tag" ||
    die "conflicts: fix them, 'git commit', then 'inkling/inkling.sh check'"
  echo "$tag" > inkling/UPSTREAM && git add inkling/UPSTREAM && { git diff --cached --quiet || git commit -q -m "inkling: upstream is Unsloth $tag"; }
}

cmd_build() {
  echo "==> build (static, Metal) -> $BIN"
  mkdir -p "$OUT" && echo '*' > "$OUT/.gitignore"
  cmake_build . "$OUT" llama-server llama-cli llama-bench llama-perplexity test-backend-ops
  mkdir -p "$BIN"
  # rename into place (new inode), so a running llama-server is never overwritten in place
  for b in llama-server llama-cli llama-bench llama-perplexity; do
    cp "$OUT/bin/$b" "$BIN/.$b.new" && mv -f "$BIN/.$b.new" "$BIN/$b"
  done
  echo "$(cat inkling/UPSTREAM) + $(git rev-parse --short HEAD)" > "$BIN/VERSION"
  echo "==> $BIN: $(cat "$BIN/VERSION")"
}

cmd_check() {
  [ -x "$OUT/bin/test-backend-ops" ] || cmd_build
  echo "==> new ops, Metal vs CPU"
  INK_EVAL=1 "$OUT/bin/test-backend-ops" test -b MTL0 2>/dev/null | grep -E 'tests passed|FAIL'
  [ -n "${MODEL:-}" ] || { echo "(set MODEL=<path to the gguf> for the speed and KLD checks)"; return; }
  local up stock=$OUT/stock text=${TEXT:-$OUT/kld.txt} chunks=${CHUNKS:-32} ref
  up=$(cat inkling/UPSTREAM)
  # default text: this repo's own docs, frozen on first use so the CPU reference stays valid across releases
  [ -s "$text" ] || { cat README.md docs/*.md docs/*/*.md tools/*/README.md 2>/dev/null || true; } | head -c 400000 > "$text"
  ref=$OUT/cpu-$(basename "$MODEL" .gguf)-$chunks.kld
  if [ "$(cat "$stock/VERSION" 2>/dev/null)" != "$up" ]; then
    echo "==> stock build of $up"
    rm -rf "$stock/src" && git worktree add -f --detach "$stock/src" "unsloth-$up" >/dev/null 2>&1
    cmake_build "$stock/src" "$stock/build" llama-bench llama-perplexity
    echo "$up" > "$stock/VERSION"
  fi
  local ppl=(-m "$MODEL" -f "$text" -c 512 --chunks "$chunks" -fa on)
  if [ ! -s "$ref" ]; then
    echo "==> CPU reference ($chunks chunks, slow)"
    "$stock/build/bin/llama-perplexity" "${ppl[@]}" -dev none -t 8 --kl-divergence-base "$ref" >/dev/null 2>&1
  fi
  echo "==> KLD vs CPU, lower is better (patched must not be above stock)"
  for b in "$stock/build/bin" "$OUT/bin"; do
    printf '%-8s ' "$([ "$b" = "$OUT/bin" ] && echo patched || echo stock)"
    "$b/llama-perplexity" "${ppl[@]}" --kl-divergence-base "$ref" --kl-divergence 2>&1 |
      grep -E '^(Mean +KLD|Same top p)' | tr -s ' ' | tr '\n' ' '; echo
  done
  echo "==> decode speed, tg64 (alternating, 2 rounds)"
  for r in 1 2; do for b in "$stock/build/bin" "$OUT/bin"; do
    printf '%-8s ' "$([ "$b" = "$OUT/bin" ] && echo patched || echo stock)"
    "$b/llama-bench" -m "$MODEL" -fa 1 -p 0 -n 64 -r 3 2>&1 | grep tg64 | awk -F'|' '{print $(NF-1), "tok/s"}'
  done; done
}

cmd_install() {
  case ${1:-} in
    studio)
      cmd_build
      cat <<EOF

Unsloth Studio (it only takes this from its own window):
  1. Settings -> llama.cpp folder -> $BIN
  2. Model settings -> speculative decoding -> Off (DSpark is slower with these patches)
  3. Reload the model.
Undo: clear the llama.cpp folder setting.
EOF
      ;;
    server)
      [ -n "${MODEL:-}" ] || die "set MODEL=<path to the gguf>"
      cmd_build
      local port=${PORT:-8080} host=${HOST:-127.0.0.1}
      mkdir -p "$(dirname "$PLIST")" "$HOME/Library/Logs" "$(dirname "$KEYFILE")"
      if [ ! -s "$KEYFILE" ]; then # one random key, readable only by you; delete the file and reinstall to rotate
        (umask 077; echo "sk-inkling-$(openssl rand -hex 32)" > "$KEYFILE")
      fi
      local dir scheme=http tls=() ca=()
      dir=$(dirname "$KEYFILE")
      if [ "$host" != 127.0.0.1 ]; then # off this Mac: HTTPS, so the key and the prompts never cross the network in clear
        scheme=https
        if [ ! -s "$dir/tls.pem" ]; then # self-signed, made once; delete tls.* and reinstall to renew
          printf '[req]\ndistinguished_name=dn\n[dn]\n[v3]\nsubjectAltName=DNS:%s.local,DNS:localhost,IP:127.0.0.1,IP:%s\nbasicConstraints=critical,CA:true\nkeyUsage=critical,digitalSignature,keyCertSign\nextendedKeyUsage=serverAuth\nsubjectKeyIdentifier=hash\nauthorityKeyIdentifier=keyid:always\n' \
            "$(scutil --get LocalHostName)" "$(ipconfig getifaddr en0 || ipconfig getifaddr en1 || echo 127.0.0.1)" > "$dir/tls.cnf"
          (umask 077; openssl ecparam -genkey -name prime256v1 -noout -out "$dir/tls.key")
          openssl req -x509 -new -key "$dir/tls.key" -days 825 -subj /CN=llama.cpp-inkling -config "$dir/tls.cnf" -extensions v3 -out "$dir/tls.pem"
        fi
        tls=(--ssl-key-file "$dir/tls.key" --ssl-cert-file "$dir/tls.pem")
        ca=(--cacert "$dir/tls.pem")
      fi
      cat > "$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key><array>
    <string>$BIN/llama-server</string><string>-m</string><string>$MODEL</string>
    <string>--host</string><string>$host</string><string>--port</string><string>$port</string>
    <string>--api-key-file</string><string>$KEYFILE</string>$(for a in ${tls[@]+"${tls[@]}"}; do printf '<string>%s</string>' "$a"; done)
    <string>--alias</string><string>${ALIAS:-$(basename "$MODEL" .gguf)}</string>
    <string>-fa</string><string>on</string><string>-c</string><string>${CTX:-65536}</string>
    <string>--parallel</string><string>4</string><string>--kv-unified</string><string>--no-context-shift</string>
    <string>--jinja</string><string>--chat-template-kwargs</string><string>{"reasoning_effort": "${EFFORT:-high}"}</string>
  </array>
  <key>RunAtLoad</key><true/><key>KeepAlive</key><true/>
  <key>StandardErrorPath</key><string>$HOME/Library/Logs/$LABEL.log</string>
</dict></plist>
EOF
      launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
      for _ in $(seq 20); do launchctl print "gui/$(id -u)/$LABEL" >/dev/null 2>&1 || break; sleep 0.5; done # bootout is async
      launchctl bootstrap "gui/$(id -u)" "$PLIST"
      echo "==> waiting for the model to load"
      for _ in $(seq 120); do curl -fs ${ca[@]+"${ca[@]}"} "$scheme://127.0.0.1:$port/health" >/dev/null 2>&1 && break; sleep 1; done
      curl -fs ${ca[@]+"${ca[@]}"} "$scheme://127.0.0.1:$port/health" >/dev/null || die "not up, see ~/Library/Logs/$LABEL.log"
      echo "OpenAI-compatible API, starts at login (log: ~/Library/Logs/$LABEL.log)"
      echo "  this Mac:     $scheme://127.0.0.1:$port/v1"
      if [ "$host" != 127.0.0.1 ]; then
        local ip; ip=$(ipconfig getifaddr en0 || ipconfig getifaddr en1 || true)
        echo "  your network: https://${ip:-<this Mac IP>}:$port/v1  or  https://$(scutil --get LocalHostName).local:$port/v1"
        echo "  certificate:  $dir/tls.pem (public; clients trust this file)"
        echo "                $(openssl x509 -noout -fingerprint -sha256 -in "$dir/tls.pem")"
      fi
      echo "  API key:      cat $KEYFILE"
      ;;
    *) die "usage: inkling.sh install studio|server" ;;
  esac
}

cmd_uninstall() {
  [ "${1:-}" = server ] || die "usage: inkling.sh uninstall server"
  launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
  rm -f "$PLIST"
  echo "==> removed $LABEL"
}

case ${1:-} in
  sync|build|check|install|uninstall) c=$1; shift; "cmd_$c" "$@" ;;
  *) sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//'; exit 1 ;;
esac
