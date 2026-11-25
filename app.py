import streamlit as st
import numpy as np
import json
import os
import time
from io import BytesIO
from PIL import Image
from sklearn.ensemble import RandomForestRegressor

# ==============================================================================
# 1. KONFIGURASI HALAMAN & GAYA UI
# ==============================================================================
st.set_page_config(
    page_title="Konstruksi S-Box ML (Non-Polinomial)",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="collapsed"
)

def load_css(file_name: str):
    css_path = os.path.join(os.path.dirname(__file__), file_name)
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("style.css")

# ==============================================================================
# 2. MODUL EVALUATOR KRIPTOGRAFI (TANPA POLINOMIAL & BEBAS OVERFLOW)
# ==============================================================================
def fwht(a: np.ndarray) -> np.ndarray:
    """Fast Walsh-Hadamard Transform (vectorized) untuk array panjang 256."""
    x = a.astype(np.int32).copy()
    h = 1
    n = len(x)
    while h < n:
        x = x.reshape(-1, 2 * h)
        u = x[:, :h].copy()
        v = x[:, h:].copy()
        x[:, :h] = u + v
        x[:, h:] = u - v
        x = x.reshape(n)
        h *= 2
    return x

def calc_nonlinearity(sbox) -> int:
    """Menghitung Nonlinearity (NL) minimum dari S-Box 8x8."""
    sbox_arr = np.array(sbox, dtype=np.int32)
    min_nl = 256
    # Evaluasi seluruh kombinasi linear tak-nol (1..255) dari 8 fungsi koordinat
    x_idx = np.arange(256, dtype=np.int32)
    for mask in range(1, 256):
        # Paritas bit dari (sbox_arr & mask)
        masked = sbox_arr & mask
        f_bit = np.zeros(256, dtype=np.int32)
        for b in range(8):
            f_bit ^= (masked >> b) & 1
        polarity = 1 - 2 * f_bit  # Aman dari uint8 overflow
        wht = fwht(polarity)
        nl = (256 - int(np.max(np.abs(wht)))) // 2
        if nl < min_nl:
            min_nl = nl
    return int(min_nl)

def calc_nonlinearity_fast(sbox) -> int:
    """Versi cepat NL (mengevaluasi 8 fungsi koordinat dasar) untuk tahap screening."""
    sbox_arr = np.array(sbox, dtype=np.int32)
    min_nl = 256
    for bit in range(8):
        f_bit = (sbox_arr >> bit) & 1
        polarity = 1 - 2 * f_bit
        wht = fwht(polarity)
        nl = (256 - int(np.max(np.abs(wht)))) // 2
        if nl < min_nl:
            min_nl = nl
    return int(min_nl)

def calc_differential_uniformity(sbox) -> int:
    """Menghitung Differential Uniformity maksimum (DDT max) secara vektorisasi."""
    sbox_arr = np.array(sbox, dtype=np.int32)
    x = np.arange(256, dtype=np.int32)
    max_du = 0
    for dx in range(1, 256):
        dy = sbox_arr[x ^ dx] ^ sbox_arr[x]
        counts = np.bincount(dy, minlength=256)
        c_max = int(np.max(counts))
        if c_max > max_du:
            max_du = c_max
    return max_du

def calc_sac(sbox) -> tuple:
    """Menghitung Strict Avalanche Criterion (SAC) rata-rata dan deviasi maksimum."""
    sbox_arr = np.array(sbox, dtype=np.int32)
    x = np.arange(256, dtype=np.int32)
    sac_matrix = np.zeros((8, 8), dtype=np.float64)
    for i in range(8):
        dx = 1 << i
        diff = sbox_arr[x ^ dx] ^ sbox_arr[x]
        for j in range(8):
            sac_matrix[i, j] = np.mean((diff >> j) & 1)
    sac_mean = float(np.mean(sac_matrix))
    sac_max_dev = float(np.max(np.abs(sac_matrix - 0.5)))
    return sac_mean, sac_max_dev

def is_bijective(sbox) -> bool:
    return len(sbox) == 256 and len(set(sbox)) == 256

def build_inverse_sbox(sbox) -> list:
    inv = [0] * 256
    for x, y in enumerate(sbox):
        inv[y] = x
    return inv

# ==============================================================================
# 3. KONSTRUKSI S-BOX BERBASIS MACHINE LEARNING (SURROGATE + GUIDED SWAP)
# ==============================================================================
def generate_chaotic_sbox(rng: np.random.Generator) -> list:
    """Membangkitkan S-Box bijektif tanpa polinomial menggunakan Tent/Logistic Map + argsort."""
    r = rng.uniform(3.96, 3.9999)
    x = rng.uniform(0.05, 0.95)
    seq = np.empty(256, dtype=np.float64)
    # Burn-in 50 iterasi agar masuk zona chaos penuh
    for _ in range(50):
        x = r * x * (1.0 - x)
    for i in range(256):
        x = r * x * (1.0 - x)
        seq[i] = x
    return np.argsort(seq).tolist()

def extract_features(sbox) -> np.ndarray:
    """Ekstraksi fitur statistik & korelasi XOR dari kandidat S-Box untuk input ML."""
    arr = np.array(sbox, dtype=np.int32)
    feats = []
    # 1. Autokorelasi XOR untuk beberapa pergeseran
    for shift in [1, 2, 4, 8, 16, 32, 64, 128]:
        xor_vals = arr ^ np.roll(arr, shift)
        hw = sum(((xor_vals >> b) & 1) for b in range(8))
        feats.extend([np.mean(hw), np.std(hw), np.min(hw), np.max(hw)])
    # 2. Distribusi bit per bit-plane & keseimbangan kuadran
    for b in range(8):
        plane = (arr >> b) & 1
        feats.append(np.mean(plane[:128]))
        feats.append(np.mean(plane[::2]))
    # 3. Selisih absolut bertetangga
    diffs = np.abs(np.diff(arr))
    feats.extend([np.mean(diffs), np.std(diffs)])
    return np.array(feats, dtype=np.float32)

def quality_score(sbox) -> float:
    """Skor kualitas gabungan untuk target regresi ML dan optimasi."""
    nl = calc_nonlinearity_fast(sbox)
    du = calc_differential_uniformity(sbox)
    _, sac_dev = calc_sac(sbox)
    return (nl * 100.0) - (du * 12.0) - (sac_dev * 500.0)

def run_ml_sbox_pipeline(n_train=400, n_screen=4000, opt_steps=600, seed=42, progress_bar=None):
    """Pipeline lengkap: Dataset Chaos -> Latih Random Forest -> Screening Cepat -> Guided Swap."""
    rng = np.random.default_rng(seed)
    t0 = time.time()

    # Tahap 1: Bangkitkan dataset pelatihan awal
    X_train, y_train, pool = [], [], []
    for i in range(n_train):
        cand = generate_chaotic_sbox(rng) if i % 2 == 0 else rng.permutation(256).tolist()
        X_train.append(extract_features(cand))
        y_train.append(quality_score(cand))
        pool.append(cand)
        if progress_bar and i % 40 == 0:
            progress_bar.progress(min(0.35, (i / n_train) * 0.35), text="Tahap 1/3: Membangkitkan dataset pelatihan awal...")

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    # Tahap 2: Latih Model Surrogate Machine Learning (Random Forest)
    if progress_bar:
        progress_bar.progress(0.45, text="Tahap 2/3: Melatih model Random Forest Regressor & menyaring ribuan kandidat...")
    rf = RandomForestRegressor(n_estimators=80, max_depth=10, random_state=seed, n_jobs=-1)
    rf.fit(X_train, y_train)

    # Screening cepat ribuan kandidat baru menggunakan prediksi ML
    t_screen_start = time.time()
    candidates = [generate_chaotic_sbox(rng) if i % 2 == 0 else rng.permutation(256).tolist() for i in range(n_screen)]
    X_screen = np.array([extract_features(c) for c in candidates])
    preds = rf.predict(X_screen)
    screen_time = time.time() - t_screen_start

    # Ambil Top-15 terbaik hasil prediksi ML untuk dievaluasi eksak
    top_indices = np.argsort(preds)[-15:]
    best_sbox = None
    best_score = -1e9
    for idx in top_indices:
        sc = quality_score(candidates[idx])
        if sc > best_score:
            best_score = sc
            best_sbox = candidates[idx].copy()

    # Tahap 3: Guided Swap Optimization (mempertahankan sifat bijektif 100%)
    history = [best_score]
    curr_sbox = best_sbox.copy()
    curr_score = best_score

    for step in range(opt_steps):
        i, j = rng.choice(256, size=2, replace=False)
        curr_sbox[i], curr_sbox[j] = curr_sbox[j], curr_sbox[i]
        new_score = quality_score(curr_sbox)
        if new_score >= curr_score:
            curr_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_sbox = curr_sbox.copy()
        else:
            # Batalkan swap jika skor menurun
            curr_sbox[i], curr_sbox[j] = curr_sbox[j], curr_sbox[i]

        if step % 15 == 0:
            history.append(best_score)
        if progress_bar and step % 50 == 0:
            prog = 0.55 + (step / opt_steps) * 0.45
            progress_bar.progress(min(1.0, prog), text=f"Tahap 3/3: Optimasi Guided Swap ({step}/{opt_steps})...")

    total_time = time.time() - t0
    if progress_bar:
        progress_bar.progress(1.0, text="Konstruksi S-Box selesai!")

    return best_sbox, history, screen_time, total_time

# ==============================================================================
# 4. FUNGSI ENKRIPSI & DEKRIPSI STANDAR KELAS (KOMPATIBEL DENGAN COLAB)
# ==============================================================================
def encrypt_with_key(data_bytes: bytes, key_bytes: bytes, sbox: list) -> bytes:
    """Enkripsi: (byte ^ key_byte) -> substitusi S-Box."""
    k_len = len(key_bytes)
    return bytes([sbox[data_bytes[i] ^ key_bytes[i % k_len]] for i in range(len(data_bytes))])

def decrypt_with_key(cipher_bytes: bytes, key_bytes: bytes, inv_sbox: list) -> bytes:
    """Dekripsi: Inverse S-Box -> (hasil ^ key_byte)."""
    k_len = len(key_bytes)
    return bytes([inv_sbox[cipher_bytes[i]] ^ key_bytes[i % k_len] for i in range(len(cipher_bytes))])

def render_sbox_html(sbox: list) -> str:
    html = "<table class='sbox-table'><thead><tr><th></th>"
    for c in range(16):
        html += f"<th>{c:02X}</th>"
    html += "</tr></thead><tbody>"
    for r in range(16):
        html += f"<tr><th>{r:02X}</th>"
        for c in range(16):
            val = sbox[r * 16 + c]
            html += f"<td>{val:02X}</td>"
        html += "</tr>"
    html += "</tbody></table>"
    return html

# ==============================================================================
# 5. INISIALISASI SESSION STATE (DEFAULT S-BOX AWAL)
# ==============================================================================
if "sbox" not in st.session_state:
    rng_init = np.random.default_rng(42)
    init_sbox = generate_chaotic_sbox(rng_init)
    st.session_state.sbox = init_sbox
    st.session_state.inv_sbox = build_inverse_sbox(init_sbox)
    st.session_state.nl = calc_nonlinearity(init_sbox)
    st.session_state.du = calc_differential_uniformity(init_sbox)
    st.session_state.sac, st.session_state.sac_dev = calc_sac(init_sbox)
    st.session_state.history = []
    st.session_state.screen_time = 0.08
    st.session_state.is_ml_optimized = False

# ==============================================================================
# 6. HEADER & KPI SUMMARY CARDS
# ==============================================================================
st.title("🔐 Konstruksi S-Box 8-bit Berbasis Machine Learning")
st.caption("Pendekatan Non-Polinomial (Chaotic Surrogate Random Forest + Guided Swap Optimization) | Kompatibel Standar Pertukaran JSON Kelas")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Nonlinearity (NL)</div>
        <div class="metric-value">{st.session_state.nl}</div>
        <div class="metric-sub">Ideal Non-Polinomial: ≥ 100</div>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Differential Uniformity (DU)</div>
        <div class="metric-value">{st.session_state.du}</div>
        <div class="metric-sub">Semakin rendah semakin tahan</div>
    </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Strict Avalanche (SAC)</div>
        <div class="metric-value">{st.session_state.sac:.4f}</div>
        <div class="metric-sub">Deviasi Maks: {st.session_state.sac_dev:.4f}</div>
    </div>
    """, unsafe_allow_html=True)
with col4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Status Bijektif & Screening</div>
        <div class="metric-value">{"VALID" if is_bijective(st.session_state.sbox) else "INVALID"}</div>
        <div class="metric-sub">ML Screening: {st.session_state.screen_time:.2f}s</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ==============================================================================
# 7. NAVIGASI 4 TAB UTAMA
# ==============================================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "1️⃣ Generator S-Box ML & Evaluasi",
    "2️⃣ Enkripsi & Dekripsi Teks",
    "3️⃣ Enkripsi & Dekripsi Gambar",
    "4️⃣ Dekripsi Ciphertext Teman"
])

# ------------------------------------------------------------------------------
# TAB 1: GENERATOR S-BOX ML & EVALUASI
# ------------------------------------------------------------------------------
with tab1:
    c_left, c_right = st.columns([1, 2])
    with c_left:
        st.subheader("⚙️ Parameter Konstruksi ML")
        seed_val = st.number_input("Random Seed", min_value=1, max_value=99999, value=42, step=1)
        n_train = st.slider("Jumlah Sampel Latih Random Forest", 200, 1000, 400, step=100)
        n_screen = st.slider("Kandidat Screening ML", 1000, 10000, 4000, step=1000)
        opt_steps = st.slider("Langkah Optimasi Guided Swap", 200, 2000, 600, step=200)

        if st.button("🚀 Jalankan Konstruksi S-Box ML", type="primary", use_container_width=True):
            pbar = st.progress(0, text="Memulai pipeline Machine Learning...")
            best_sb, hist, s_time, t_time = run_ml_sbox_pipeline(
                n_train=n_train, n_screen=n_screen, opt_steps=opt_steps, seed=int(seed_val), progress_bar=pbar
            )
            st.session_state.sbox = best_sb
            st.session_state.inv_sbox = build_inverse_sbox(best_sb)
            st.session_state.nl = calc_nonlinearity(best_sb)
            st.session_state.du = calc_differential_uniformity(best_sb)
            st.session_state.sac, st.session_state.sac_dev = calc_sac(best_sb)
            st.session_state.history = hist
            st.session_state.screen_time = s_time
            st.session_state.is_ml_optimized = True
            st.rerun()

        st.markdown("### 📥 Ekspor File S-Box")
        sbox_json_str = json.dumps({
            "method": "ML-Surrogate-Chaotic-NonPolynomial",
            "metrics": {
                "nonlinearity": st.session_state.nl,
                "differential_uniformity": st.session_state.du,
                "sac": round(st.session_state.sac, 6)
            },
            "sbox": st.session_state.sbox
        }, indent=2)

        st.download_button(
            label="⬇️ Download my_sbox.json (Untuk Tukar ke Teman)",
            data=sbox_json_str,
            file_name="my_sbox.json",
            mime="application/json",
            use_container_width=True
        )

        txt_lines = []
        for r in range(16):
            txt_lines.append(" ".join(str(v) for v in st.session_state.sbox[r*16:(r+1)*16]))
        st.download_button(
            label="⬇️ Download my_sbox.txt (Format Matriks)",
            data="\n".join(txt_lines),
            file_name="my_sbox.txt",
            mime="text/plain",
            use_container_width=True
        )

    with c_right:
        st.subheader("📊 Tabel Substitusi S-Box 8×8 (Heksadesimal)")
        st.markdown(render_sbox_html(st.session_state.sbox), unsafe_allow_html=True)

        if st.session_state.history:
            st.subheader("📈 Progres Optimasi Skor Kriptografi (Guided Swap)")
            st.line_chart(st.session_state.history)

# ------------------------------------------------------------------------------
# TAB 2: ENKRIPSI & DEKRIPSI TEKS
# ------------------------------------------------------------------------------
with tab2:
    st.subheader("📝 Uji Enkripsi & Dekripsi Plainteks Teks")
    if "text_key_hex" not in st.session_state:
        st.session_state.text_key_hex = os.urandom(16).hex().upper()

    col_k1, col_k2 = st.columns([3, 1])
    with col_k1:
        key_hex_input = st.text_input("Key 128-bit (32 Karakter Hex)", value=st.session_state.text_key_hex)
    with col_k2:
        st.write("")
        st.write("")
        if st.button("🎲 Generate Key Baru", use_container_width=True):
            st.session_state.text_key_hex = os.urandom(16).hex().upper()
            st.rerun()

    plaintext_in = st.text_area(
        "Masukkan Plainteks:",
        value="Pengujian konstruksi S-Box 8-bit berbasis Machine Learning tanpa polinomial berhasil dilakukan!",
        height=110
    )

    if st.button("🔒 Enkripsi & Verifikasi Dekripsi Teks", type="primary"):
        try:
            key_bytes = bytes.fromhex(key_hex_input.strip())
            if len(key_bytes) == 0:
                raise ValueError("Key tidak boleh kosong.")
            plain_bytes = plaintext_in.encode("utf-8")
            cipher_bytes = encrypt_with_key(plain_bytes, key_bytes, st.session_state.sbox)
            decrypted_str = decrypt_with_key(cipher_bytes, key_bytes, st.session_state.inv_sbox).decode("utf-8")

            st.success(f"Status Kecocokan Dekripsi: {'COCOK 100% (BERHASIL)' if decrypted_str == plaintext_in else 'GAGAL'}")
            st.text_area("Ciphertext (Hex) — Salin ini untuk dikirim ke teman:", value=cipher_bytes.hex().upper(), height=110)
            st.text_area("Hasil Dekripsi Kembali:", value=decrypted_str, height=90)
        except Exception as e:
            st.error(f"Terjadi kesalahan format Key/Input: {e}")

# ------------------------------------------------------------------------------
# TAB 3: ENKRIPSI & DEKRIPSI GAMBAR
# ------------------------------------------------------------------------------
with tab3:
    st.subheader("🖼️ Uji Enkripsi & Dekripsi File Gambar")
    uploaded_img = st.file_uploader("Upload Gambar (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"])

    if uploaded_img is not None:
        img_bytes = uploaded_img.read()
        if "img_key_hex" not in st.session_state:
            st.session_state.img_key_hex = os.urandom(16).hex().upper()

        c_ik1, c_ik2 = st.columns([3, 1])
        with c_ik1:
            img_key_in = st.text_input("Key Gambar 128-bit (Hex)", value=st.session_state.img_key_hex)
        with c_ik2:
            st.write("")
            st.write("")
            if st.button("🎲 Acak Key Gambar", use_container_width=True):
                st.session_state.img_key_hex = os.urandom(16).hex().upper()
                st.rerun()

        if st.button("🔐 Proses Enkripsi & Dekripsi Gambar", type="primary"):
            try:
                k_img = bytes.fromhex(img_key_in.strip())
                cipher_img_bytes = encrypt_with_key(img_bytes, k_img, st.session_state.sbox)
                dec_img_bytes = decrypt_with_key(cipher_img_bytes, k_img, st.session_state.inv_sbox)

                st.success(f"Integritas Byte Gambar: {'IDENTIK 100% (BERHASIL)' if dec_img_bytes == img_bytes else 'GAGAL'} ({len(img_bytes):,} byte)")
                st.code(f"Preview Ciphertext Hex (64 byte awal): {cipher_img_bytes[:64].hex().upper()}")

                col_g1, col_g2, col_g3 = st.columns(3)
                with col_g1:
                    st.image(img_bytes, caption="1. Gambar Asli (Plaintext)", use_container_width=True)
                with col_g2:
                    # Visualisasi noise dari byte terenkripsi
                    orig_pil = Image.open(BytesIO(img_bytes)).convert("RGB")
                    w, h = orig_pil.size
                    raw_pixels = np.frombuffer(cipher_img_bytes, dtype=np.uint8)
                    needed = w * h * 3
                    if len(raw_pixels) < needed:
                        raw_pixels = np.pad(raw_pixels, (0, needed - len(raw_pixels)), mode="wrap")
                    noise_img = Image.fromarray(raw_pixels[:needed].reshape((h, w, 3)))
                    st.image(noise_img, caption="2. Visualisasi Noise Ciphertext", use_container_width=True)
                with col_g3:
                    st.image(dec_img_bytes, caption="3. Hasil Dekripsi (Inv S-Box)", use_container_width=True)

                dl1, dl2 = st.columns(2)
                with dl1:
                    st.download_button(
                        "⬇️ Download File Ciphertext (.bin)",
                        data=cipher_img_bytes,
                        file_name=f"encrypted_{uploaded_img.name}.bin",
                        mime="application/octet-stream",
                        use_container_width=True
                    )
                with dl2:
                    st.download_button(
                        "⬇️ Download Gambar Hasil Dekripsi",
                        data=dec_img_bytes,
                        file_name=f"decrypted_{uploaded_img.name}",
                        mime="image/png",
                        use_container_width=True
                    )
            except Exception as e:
                st.error(f"Gagal memproses gambar: {e}")

# ------------------------------------------------------------------------------
# TAB 4: DEKRIPSI CIPHERTEXT MILIK TEMAN
# ------------------------------------------------------------------------------
with tab4:
    st.subheader("🤝 Dekripsi Pesan / File Milik Teman")
    st.info("Unggah file `my_sbox.json` milik temanmu, lalu masukkan Key (Hex) dan Ciphertext (Hex) yang dikirimkan.")

    friend_json_file = st.file_uploader("1. Upload file S-Box JSON teman (my_sbox.json)", type=["json"])
    friend_key_hex = st.text_input("2. Tempel Key (Hex) dari teman:").strip()
    friend_cipher_hex = st.text_area("3. Tempel Ciphertext (Hex) dari teman:", height=120).strip()

    if st.button("🔓 Dekripsi Pesan Teman", type="primary"):
        if not friend_json_file or not friend_key_hex or not friend_cipher_hex:
            st.warning("Pastikan file JSON S-Box, Key (Hex), dan Ciphertext (Hex) sudah diisi semua.")
        else:
            try:
                friend_data = json.load(friend_json_file)
                friend_sbox = friend_data["sbox"] if isinstance(friend_data, dict) else friend_data
                if not is_bijective(friend_sbox):
                    st.error("S-Box di dalam file JSON tidak valid (harus 256 elemen unik 0..255).")
                else:
                    friend_inv = build_inverse_sbox(friend_sbox)
                    f_key = bytes.fromhex(friend_key_hex)
                    f_cipher = bytes.fromhex(friend_cipher_hex)
                    hasil_bytes = decrypt_with_key(f_cipher, f_key, friend_inv)

                    try:
                        decoded_msg = hasil_bytes.decode("utf-8")
                        st.success("Dekripsi Teks Berhasil!")
                        st.text_area("Hasil Dekripsi Pesan:", value=decoded_msg, height=120)
                    except UnicodeDecodeError:
                        st.success("Dekripsi Biner Berhasil! (File berupa gambar/biner non-UTF8)")
                        st.download_button(
                            "⬇️ Download Hasil Dekripsi (hasil_teman.bin)",
                            data=hasil_bytes,
                            file_name="hasil_teman.bin",
                            mime="application/octet-stream"
                        )
            except Exception as e:
                st.error(f"Gagal mendekripsi data teman: {e}")
