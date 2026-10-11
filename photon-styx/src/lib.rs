//! `libphotonstyx.so`: PhotonVision's experimental Styx capture and Eidos AprilTag backend,
//! as JNI for `org.photonvision.styx.StyxJNI`.
//!
//! Handles are positive `long`s into a process-wide table. Each JNI entry point catches panics
//! and turns them into Java exceptions (a panic must never unwind into the JVM).

pub mod config;
pub mod detect;
pub mod geometry;
pub mod overlay;
pub mod pack;
pub mod session;

#[cfg(test)]
mod pose_tests;

use std::collections::HashMap;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::{Arc, LazyLock, Mutex, MutexGuard};
use std::time::Duration;

use jni::JNIEnv;
use jni::objects::{JByteArray, JClass, JObjectArray, JString};
use jni::sys::{jdouble, jint, jlong};

use config::PipelineConfig;
use session::Session;

static SESSIONS: LazyLock<Mutex<HashMap<i64, Arc<Session>>>> = LazyLock::new(|| Mutex::new(HashMap::new()));
static NEXT_HANDLE: AtomicI64 = AtomicI64::new(1);

fn sessions() -> MutexGuard<'static, HashMap<i64, Arc<Session>>> {
    SESSIONS.lock().unwrap_or_else(|e| e.into_inner())
}

fn session(handle: i64) -> Result<Arc<Session>, String> {
    sessions()
        .get(&handle)
        .cloned()
        .ok_or_else(|| format!("no open camera for handle {handle}"))
}

/// Run `body`, turning an error into a Java exception of `exception` and a panic into an
/// `IllegalStateException`. Returns `fallback` on either.
fn guarded<'l, T>(
    env: &mut JNIEnv<'l>,
    exception: &str,
    fallback: T,
    body: impl FnOnce(&mut JNIEnv<'l>) -> Result<T, String>,
) -> T {
    let outcome = catch_unwind(AssertUnwindSafe(|| body(env)));
    match outcome {
        Ok(Ok(value)) => value,
        Ok(Err(message)) => {
            let _ = env.throw_new(exception, message);
            fallback
        }
        Err(_) => {
            let _ = env.throw_new("java/lang/IllegalStateException", "photonstyx panicked");
            fallback
        }
    }
}

fn millis(ms: jlong) -> Duration {
    Duration::from_millis(ms.max(0) as u64)
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_photonvision_styx_StyxJNI_listCameras<'l>(
    mut env: JNIEnv<'l>,
    _class: JClass<'l>,
) -> JObjectArray<'l> {
    guarded(&mut env, "java/lang/IllegalStateException", JObjectArray::default(), |env| {
        let names = session::list_cameras();
        let string_class = env.find_class("java/lang/String").map_err(|e| e.to_string())?;
        let array = env
            .new_object_array(names.len() as i32, string_class, JString::default())
            .map_err(|e| e.to_string())?;
        for (i, name) in names.iter().enumerate() {
            let js = env.new_string(name).map_err(|e| e.to_string())?;
            env.set_object_array_element(&array, i as i32, js).map_err(|e| e.to_string())?;
        }
        Ok(array)
    })
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_photonvision_styx_StyxJNI_open<'l>(
    mut env: JNIEnv<'l>,
    _class: JClass<'l>,
    camera: JString<'l>,
    width: jint,
    height: jint,
    fps: jint,
) -> jlong {
    guarded(&mut env, "java/lang/IllegalStateException", 0, |env| {
        let camera: String = env.get_string(&camera).map_err(|e| e.to_string())?.into();
        let opened = Session::open(&camera, width.max(1) as u32, height.max(1) as u32, fps.max(0) as u32)?;
        let handle = NEXT_HANDLE.fetch_add(1, Ordering::Relaxed);
        sessions().insert(handle, Arc::new(opened));
        Ok(handle)
    })
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_photonvision_styx_StyxJNI_close(
    mut env: JNIEnv<'_>,
    _class: JClass<'_>,
    handle: jlong,
) {
    guarded(&mut env, "java/lang/IllegalStateException", (), |_| {
        let removed = sessions().remove(&handle);
        if let Some(session) = removed {
            session.stop();
        }
        Ok(())
    })
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_photonvision_styx_StyxJNI_configure(
    mut env: JNIEnv<'_>,
    _class: JClass<'_>,
    handle: jlong,
    json: JString<'_>,
) {
    guarded(&mut env, "java/lang/IllegalArgumentException", (), |env| {
        let json: String = env.get_string(&json).map_err(|e| e.to_string())?.into();
        let config = PipelineConfig::parse(&json)?;
        session(handle)?.configure(&config)
    })
}

/// Apply a standard camera control (`session::CONTROL_*`); returns the value in effect.
#[unsafe(no_mangle)]
pub extern "system" fn Java_org_photonvision_styx_StyxJNI_setControl<'l>(
    mut env: JNIEnv<'l>,
    _class: JClass<'l>,
    handle: jlong,
    kind: jint,
    value: jdouble,
) -> jdouble {
    guarded(&mut env, "java/lang/IllegalStateException", f64::NAN, |_| {
        session(handle)?.set_control(kind, value)
    })
}

/// Wait up to `timeoutMs` for a frame and run the detector on it. Returns the packed result
/// (see `pack.rs`), or an empty array on timeout.
#[unsafe(no_mangle)]
pub extern "system" fn Java_org_photonvision_styx_StyxJNI_processFrame<'l>(
    mut env: JNIEnv<'l>,
    _class: JClass<'l>,
    handle: jlong,
    timeout_ms: jlong,
) -> JByteArray<'l> {
    guarded(&mut env, "java/lang/IllegalStateException", JByteArray::default(), |env| {
        let bytes = session(handle)?.process(millis(timeout_ms))?.unwrap_or_default();
        env.byte_array_from_slice(&bytes).map_err(|e| e.to_string())
    })
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_photonvision_styx_StyxJNI_rawJpeg<'l>(
    mut env: JNIEnv<'l>,
    _class: JClass<'l>,
    handle: jlong,
    timeout_ms: jlong,
) -> JByteArray<'l> {
    guarded(&mut env, "java/lang/IllegalStateException", JByteArray::default(), |env| {
        let bytes = session(handle)?.raw_jpeg(millis(timeout_ms)).unwrap_or_default();
        env.byte_array_from_slice(&bytes).map_err(|e| e.to_string())
    })
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_photonvision_styx_StyxJNI_processedJpeg<'l>(
    mut env: JNIEnv<'l>,
    _class: JClass<'l>,
    handle: jlong,
    timeout_ms: jlong,
) -> JByteArray<'l> {
    guarded(&mut env, "java/lang/IllegalStateException", JByteArray::default(), |env| {
        let bytes = session(handle)?.processed_jpeg(millis(timeout_ms)).unwrap_or_default();
        env.byte_array_from_slice(&bytes).map_err(|e| e.to_string())
    })
}

/// CLOCK_BOOTTIME now, the clock frame capture times are on.
#[unsafe(no_mangle)]
pub extern "system" fn Java_org_photonvision_styx_StyxJNI_nowBoottimeNanos(
    _env: JNIEnv<'_>,
    _class: JClass<'_>,
) -> jlong {
    let mut ts = libc::timespec { tv_sec: 0, tv_nsec: 0 };
    // SAFETY: clock_gettime writes one timespec through a valid pointer.
    unsafe { libc::clock_gettime(libc::CLOCK_BOOTTIME, &mut ts) };
    ts.tv_sec as jlong * 1_000_000_000 + ts.tv_nsec as jlong
}
