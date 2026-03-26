use std::env;
use std::fs;
use std::path::{Path, PathBuf};

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("manifest dir"));
    let ui_dist_dir = manifest_dir.join("ui").join("dist");
    let index_html = ui_dist_dir.join("index.html");

    println!("cargo:rerun-if-changed={}", ui_dist_dir.display());

    if !index_html.is_file() {
        panic!(
            "Embedded UI assets are missing at {}. Run `cd ui && npm ci && npm run build` before cargo build or cargo run.",
            index_html.display()
        );
    }

    let mut files = Vec::new();
    collect_files(&ui_dist_dir, &mut files);
    files.sort();

    for file in &files {
        println!("cargo:rerun-if-changed={}", file.display());
    }

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));
    let generated_path = out_dir.join("ui_assets.rs");
    let generated = build_embedded_assets_module(&ui_dist_dir, &files);
    fs::write(&generated_path, generated).expect("write generated ui assets module");
}

fn collect_files(dir: &Path, files: &mut Vec<PathBuf>) {
    let entries = fs::read_dir(dir).unwrap_or_else(|error| {
        panic!(
            "Failed to read UI dist directory {}: {}",
            dir.display(),
            error
        )
    });

    for entry in entries {
        let entry = entry.unwrap_or_else(|error| panic!("Failed to read UI dist entry: {}", error));
        let path = entry.path();
        if path.is_dir() {
            collect_files(&path, files);
        } else if path.is_file() {
            files.push(path);
        }
    }
}

fn build_embedded_assets_module(ui_dist_dir: &Path, files: &[PathBuf]) -> String {
    let mut generated = String::from(
        "pub(crate) struct EmbeddedUiAsset {\n    pub(crate) path: &'static str,\n    pub(crate) bytes: &'static [u8],\n    pub(crate) content_type: &'static str,\n}\n\npub(crate) static EMBEDDED_UI_ASSETS: &[EmbeddedUiAsset] = &[\n",
    );

    for file in files {
        let relative = file
            .strip_prefix(ui_dist_dir)
            .unwrap_or_else(|error| panic!("Failed to strip UI dist prefix: {}", error))
            .to_string_lossy()
            .replace('\\', "/");
        let absolute = escape_rust_string(&file.to_string_lossy());
        let path = escape_rust_string(&relative);
        let content_type = mime_type_for_path(file);

        generated.push_str(&format!(
            "    EmbeddedUiAsset {{ path: \"{}\", bytes: include_bytes!(\"{}\"), content_type: \"{}\" }},\n",
            path, absolute, content_type
        ));
    }

    generated.push_str("];\n");
    generated
}

fn mime_type_for_path(path: &Path) -> &'static str {
    match path
        .extension()
        .and_then(|ext| ext.to_str())
        .unwrap_or_default()
    {
        "html" => "text/html; charset=utf-8",
        "css" => "text/css; charset=utf-8",
        "js" => "application/javascript; charset=utf-8",
        "json" => "application/json; charset=utf-8",
        "svg" => "image/svg+xml",
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "webp" => "image/webp",
        "ico" => "image/x-icon",
        "txt" => "text/plain; charset=utf-8",
        "map" => "application/json; charset=utf-8",
        "woff" => "font/woff",
        "woff2" => "font/woff2",
        _ => "application/octet-stream",
    }
}

fn escape_rust_string(input: &str) -> String {
    input.replace('\\', "\\\\").replace('"', "\\\"")
}
