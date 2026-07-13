mod app;
mod gpu;
mod image_io;
mod model_cache;

use app::App;

fn main() {
    console_error_panic_hook::set_once();
    let _ = console_log::init_with_level(log::Level::Info);
    leptos::mount::mount_to_body(App);
}
