use std::path::Path;
use std::process::Command;

pub(super) fn rustfmt_generated_file(path: &Path) {
    if !path.exists() {
        return;
    }

    match Command::new("rustfmt")
        .arg("--emit")
        .arg("files")
        .arg(path)
        .output()
    {
        Ok(output) if output.status.success() => {}
        Ok(output) => {
            let stderr = String::from_utf8_lossy(&output.stderr);
            let reason = if stderr.trim().is_empty() {
                format!("exit status {}", output.status)
            } else {
                stderr.replace('\n', " ").trim().to_string()
            };
            eprintln!("warning: rustfmt failed for {}: {}", path.display(), reason);
        }
        Err(err) => {
            eprintln!(
                "warning: rustfmt unavailable; skipped formatting {}: {}",
                path.display(),
                err
            );
        }
    }
}
