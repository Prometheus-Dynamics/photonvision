//! `libphotonstyx.so`: PhotonVision's experimental Styx capture and Eidos AprilTag backend,
//! as JNI for `org.photonvision.styx.StyxJNI`.
//!
//! Handles are positive `long`s into a process-wide table. Each JNI entry point catches panics
//! and turns them into Java exceptions (a panic must never unwind into the JVM).

pub mod calibrate;
pub mod config;
pub mod detect;
pub mod geometry;
pub mod overlay;
pub mod pack;
pub mod session;

#[cfg(test)]
mod e2e_tests;
#[cfg(test)]
mod pose_tests;

use std::collections::HashMap;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::{Arc, LazyLock, Mutex, MutexGuard};
use std::time::Duration;

use jni::JNIEnv;
use eidos_calibration::PixelCoordinate;
use jni::objects::{JByteArray, JClass, JDoubleArray, JIntArray, JObjectArray, JString};
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

/// Ask for a board detection (`calibrate::BoardConfig` JSON) on the next camera frame. The result
/// rides in that frame's record (`FrameRecord::board`, see `pack.rs`).
#[unsafe(no_mangle)]
pub extern "system" fn Java_org_photonvision_styx_StyxJNI_requestBoard(
    mut env: JNIEnv<'_>,
    _class: JClass<'_>,
    handle: jlong,
    board: JString<'_>,
) {
    guarded(&mut env, "java/lang/IllegalArgumentException", (), |env| {
        let json: String = env.get_string(&board).map_err(|e| e.to_string())?.into();
        let board = calibrate::BoardConfig::parse(&json)?;
        session(handle)?.request_board(board);
        Ok(())
    })
}

/// Solve the camera from board views. Views are flat: `viewSizes[v]` corners of view `v`, taken in
/// order from `ids` (chessboard or ChArUco corner ids) and `corners` (u, v pairs). Returns
/// `[fx, fy, cx, cy, skew, rms, distortionCount, converged, distortion..., viewRms...]`, the
/// distortion in OpenCV's order (k1 k2 p1 p2 k3 for `pinhole`, k1 k2 k3 k4 for `fisheye`).
#[unsafe(no_mangle)]
pub extern "system" fn Java_org_photonvision_styx_StyxJNI_calibrate<'l>(
    mut env: JNIEnv<'l>,
    _class: JClass<'l>,
    board: JString<'l>,
    lens: JString<'l>,
    view_sizes: JIntArray<'l>,
    ids: JIntArray<'l>,
    corners: JDoubleArray<'l>,
) -> JDoubleArray<'l> {
    guarded(&mut env, "java/lang/IllegalArgumentException", JDoubleArray::default(), |env| {
        let board: String = env.get_string(&board).map_err(|e| e.to_string())?.into();
        let lens: String = env.get_string(&lens).map_err(|e| e.to_string())?.into();
        let board = calibrate::BoardConfig::parse(&board)?;
        let lens = calibrate::lens_model(&lens)?;
        let sizes = int_array(env, &view_sizes)?;
        let ids = int_array(env, &ids)?;
        let corners = double_array(env, &corners)?;
        let views = split_views(&sizes, &ids, &corners)?;
        let solved = calibrate::solve(&board, lens, &views)?;

        let mut out = vec![
            solved.fx,
            solved.fy,
            solved.cx,
            solved.cy,
            solved.skew,
            solved.rms,
            solved.distortion.len() as f64,
            if solved.converged { 1.0 } else { 0.0 },
        ];
        out.extend(&solved.distortion);
        out.extend(&solved.view_rms);
        let array = env.new_double_array(out.len() as i32).map_err(|e| e.to_string())?;
        env.set_double_array_region(&array, 0, &out).map_err(|e| e.to_string())?;
        Ok(array)
    })
}

fn int_array(env: &mut JNIEnv<'_>, array: &JIntArray<'_>) -> Result<Vec<i32>, String> {
    let len = env.get_array_length(array).map_err(|e| e.to_string())? as usize;
    let mut out = vec![0i32; len];
    env.get_int_array_region(array, 0, &mut out).map_err(|e| e.to_string())?;
    Ok(out)
}

fn double_array(env: &mut JNIEnv<'_>, array: &JDoubleArray<'_>) -> Result<Vec<f64>, String> {
    let len = env.get_array_length(array).map_err(|e| e.to_string())? as usize;
    let mut out = vec![0f64; len];
    env.get_double_array_region(array, 0, &mut out).map_err(|e| e.to_string())?;
    Ok(out)
}

/// Splits the flat view arrays (see `calibrate` above) into views, checking their sizes.
fn split_views(sizes: &[i32], ids: &[i32], corners: &[f64]) -> Result<Vec<calibrate::View>, String> {
    if corners.len() != 2 * ids.len() {
        return Err(format!("{} corner coordinates for {} corner ids", corners.len(), ids.len()));
    }
    let mut views = Vec::with_capacity(sizes.len());
    let mut at = 0usize;
    for &size in sizes {
        let size = usize::try_from(size).map_err(|_| format!("bad view size {size}"))?;
        if at + size > ids.len() {
            return Err("the view sizes name more corners than were given".into());
        }
        let mut view = calibrate::View::default();
        for i in at..at + size {
            let id = u32::try_from(ids[i]).map_err(|_| format!("bad corner id {}", ids[i]))?;
            view.ids.push(id);
            view.corners.push(PixelCoordinate {
                u: corners[2 * i],
                v: corners[2 * i + 1],
            });
        }
        views.push(view);
        at += size;
    }
    if at != ids.len() {
        return Err(format!("the view sizes name {at} corners, but {} were given", ids.len()));
    }
    Ok(views)
}
