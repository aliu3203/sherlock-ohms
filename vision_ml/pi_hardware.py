"""Optional Pi camera/GPIO with deterministic laptop fallbacks."""
import logging
from pathlib import Path
import time
import cv2
from .registration import _gray

LOGGER = logging.getLogger(__name__)


class TrafficLight:
    PINS = {'red': 17, 'yellow': 27, 'green': 22}
    COLORS = {'red': '\033[31m', 'yellow': '\033[33m', 'green': '\033[32m'}

    def __init__(self, *, force_fallback=False):
        self.leds = {}
        self.state = None
        if not force_fallback:
            try:
                from gpiozero import LED
                for color, pin in self.PINS.items():
                    self.leds[color] = LED(pin)
            except Exception as exc:
                LOGGER.info('GPIO unavailable; using terminal: %s', exc)
                self.close()

    def set(self, color):
        color = color.lower()
        if color not in self.PINS:
            raise ValueError('Traffic light must be red, yellow, or green')
        for name, led in self.leds.items():
            led.on() if name == color else led.off()
        self.state = color
        if not self.leds:
            print(f'{self.COLORS[color]}[GPIO] {color.upper()}\033[0m')

    def close(self):
        for led in self.leds.values():
            led.close()
        self.leds.clear()
        self.state = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class BreadboardCamera:
    """Return the sharpest of three frames with fixed focus and exposure.

    On Pi, exposure/gain settle once, then AE/AWB are locked. Autofocus lenses
    are set to manual LensPosition. Fixed-focus cameras skip focus controls.
    Legacy default: failures use local images cyclically, with warning/provenance.
    require_live=True forbids fallback at initialization and during capture.
    """
    def __init__(self, local_images=None, *, force_fallback=False,
                 size=(1920, 1080), lens_position=1.0, warmup_seconds=1.0,
                 require_live=False):
        if require_live and force_fallback:
            raise ValueError("require_live conflicts with force_fallback")
        self.require_live = require_live
        self.fallback_reason = None
        self.last_capture = None
        self.locked_controls = {}
        self.sequence = 0
        if local_images is None:
            local_images = []
        if isinstance(local_images, (str, Path)):
            path = Path(local_images)
            local_images = sorted(path.glob('*.png')) if path.is_dir() else [path]
        self.local_images = list(map(Path, local_images))
        self.index, self.camera = 0, None
        if not force_fallback:
            try:
                from picamera2 import Picamera2
                self.camera = Picamera2()
                # Picamera2 RGB888 is BGR byte order for OpenCV consumers.
                config = self.camera.create_still_configuration(main={'size': size, 'format': 'RGB888'})
                self.camera.configure(config)
                self.camera.start()
                time.sleep(warmup_seconds)
                metadata = self.camera.capture_metadata()
                available = self.camera.camera_controls
                controls = {}
                if 'AfMode' in available:
                    controls.update(AfMode=0, LensPosition=lens_position)
                if 'AeEnable' in available:
                    controls.update(AeEnable=False, ExposureTime=metadata['ExposureTime'],
                                    AnalogueGain=metadata['AnalogueGain'])
                if 'AwbEnable' in available and 'ColourGains' in metadata:
                    controls.update(AwbEnable=False, ColourGains=metadata['ColourGains'])
                self.camera.set_controls(controls)
                self.locked_controls = controls
                # Drain queued frames so subsequent bursts use locked settings.
                for _ in range(3):
                    self.camera.capture_array('main')
            except Exception as exc:
                self.close()
                self.fallback_reason = str(exc)
                if self.require_live:
                    raise RuntimeError('Requested live camera is unavailable') from exc
                LOGGER.warning('Camera unavailable; using local images: %s', exc)

    def _local_frame(self):
        if not self.local_images:
            raise RuntimeError('No Pi camera and no fallback local images supplied')
        path = self.local_images[self.index % len(self.local_images)]
        self.index += 1
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f'Cannot read fallback image {path}')
        self._local_path = str(path.resolve())
        return image

    @property
    def source(self):
        """Current backend; last_capture records the source of the returned frame."""
        return 'pi' if self.camera is not None else 'local'

    def capture_frame(self):
        """One fresh acquisition, with visible provenance; never synthetic."""
        source = self.source
        if self.camera is not None:
            try:
                image = self.camera.capture_array('main')
            except Exception as exc:
                self.close()
                self.fallback_reason = str(exc)
                if self.require_live:
                    raise RuntimeError('Requested live camera capture failed') from exc
                LOGGER.warning('Camera capture failed; using local images: %s', exc)
                source = 'local'
                image = self._local_frame()
        else:
            if self.require_live:
                raise RuntimeError('Requested live camera is closed or unavailable')
            image = self._local_frame()
        self.sequence += 1
        self.last_capture = {
            'source': source, 'sequence': self.sequence, 'timestamp': time.time(),
            'local_path': self._local_path if source == 'local' else None,
            'locked_controls': self.locked_controls if source == 'pi' else {},
            'fallback_reason': self.fallback_reason,
        }
        return image.copy()

    def capture_burst(self):
        frames = []
        for _ in range(3):
            frame = self.capture_frame()
            frames.append((frame, self.last_capture.copy()))
        image, self.last_capture = max(
            frames, key=lambda item: float(cv2.Laplacian(_gray(item[0]), cv2.CV_64F).var()))
        return image.copy()

    capture = capture_burst

    def close(self):
        camera, self.camera = self.camera, None
        if camera is not None:
            try:
                camera.stop()
            except Exception as exc:
                LOGGER.debug('Camera stop during cleanup: %s', exc)
            try:
                camera.close()
            except Exception as exc:
                LOGGER.debug('Camera close during cleanup: %s', exc)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
