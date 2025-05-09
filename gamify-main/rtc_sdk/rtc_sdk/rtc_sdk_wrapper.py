import ctypes
import json
import os
import urllib.parse
import urllib.request

SDK_API_URL = "https://rtcrobot.com/api"
LIBAOSL_PATH = os.path.join(os.path.dirname(__file__), "lib", "libaosl.so")
LIBAGORA_RTM_SDK_PATH = os.path.join(os.path.dirname(__file__), "lib", "libagora_rtm_sdk.so")
LIBRTC_SDK_PATH = os.path.join(os.path.dirname(__file__), "lib", "librtc_sdk.so")

libaosl = ctypes.CDLL(LIBAOSL_PATH)
libagora_rtm_sdk = ctypes.CDLL(LIBAGORA_RTM_SDK_PATH, mode=ctypes.RTLD_GLOBAL)
librtc_sdk = ctypes.CDLL(LIBRTC_SDK_PATH)

MessageReceivedCallbackType = ctypes.CFUNCTYPE(None, ctypes.c_char_p)
ConnectionStateChangedCallbackType = ctypes.CFUNCTYPE(None, ctypes.c_int)

librtc_sdk.register_event_callback.argtypes = [MessageReceivedCallbackType, ConnectionStateChangedCallbackType]
librtc_sdk.register_event_callback.restype = None

librtc_sdk.connect.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p]
librtc_sdk.connect.restype = None

_global_message_received_cb = None
_global_connection_state_changed_cb = None


def register_event_callback(message_received_callback, connection_state_changed_callback):
    global _global_message_received_cb, _global_connection_state_changed_cb
    _global_message_received_cb = MessageReceivedCallbackType(message_received_callback)
    _global_connection_state_changed_cb = ConnectionStateChangedCallbackType(connection_state_changed_callback)
    librtc_sdk.register_event_callback(_global_message_received_cb, _global_connection_state_changed_cb)


def connect(secret_id, secret_key, room_id):
    user_id = room_id + "_robot"
    params = {"secret_id": secret_id, "secret_key": secret_key, "user_id": user_id}
    print(params)
    encoded_params = urllib.parse.urlencode(params)
    url = SDK_API_URL + "/room/" + room_id + "?" + encoded_params
    with urllib.request.urlopen(url) as response:
        data = response.read()
        room_info = json.loads(data)
        if room_info["rtm_token"] is None:
            raise Exception("Failed to get token")
        librtc_sdk.connect(room_id.encode(), user_id.encode(), room_info["rtm_token"].encode())


def disconnect():
    librtc_sdk.disconnect()
