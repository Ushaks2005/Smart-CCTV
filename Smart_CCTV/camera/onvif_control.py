from onvif import ONVIFCamera

def control_onvif(ip, port, username, password):
    cam = ONVIFCamera(ip, port, username, password)
    media_service = cam.create_media_service()
    ptz_service = cam.create_ptz_service()

    # Example: Move camera to preset
    presets = ptz_service.GetPresets({'ProfileToken': 'profile_1'})
    ptz_service.GotoPreset({'ProfileToken': 'profile_1', 'PresetToken': presets[0].token})
