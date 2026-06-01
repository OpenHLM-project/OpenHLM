#!/usr/bin/env python3
"""
XsensToGMR Adapter
Converts Xsens MVN real-time streaming data to GMR-compatible format for humanoid robot retargeting.
"""

import numpy as np
from typing import Dict, Tuple, Optional, List
from scipy.spatial.transform import Rotation as R
from xsens_mvn_robot import XsensWrapper


class XsensToGMR:
    """
    Adapter class to convert Xsens MVN streaming data to GMR human_frame format.

    GMR expects:
        human_frame = {
            "Body_Name": ([x, y, z], [qw, qx, qy, qz]),
            ...
        }

    Where positions are in meters (global frame) and orientations are quaternions (scalar-first).
    """
    
    # Mapping from Xsens link names to GMR expected body names
    # Based on actual Xsens MVN output (lowercase with underscores)
    XSENS_TO_GMR_MAPPING = {
        # Core body (REQUIRED)
        'pelvis': 'Pelvis',
        'l5': 'Spine',
        'l3': 'Spine1',
        't12': 'Spine2', 
        't8': 'Chest',
        'neck': 'Neck',
        'head': 'Head',
        
        # Left arm (REQUIRED: left_hand)
        'left_shoulder': 'Left_Shoulder',
        'left_upper_arm': 'Left_UpperArm',
        'left_forearm': 'Left_Forearm',
        'left_hand': 'Left_Hand',
        
        # Right arm (REQUIRED: right_hand)
        'right_shoulder': 'Right_Shoulder',
        'right_upper_arm': 'Right_UpperArm',
        'right_forearm': 'Right_Forearm',
        'right_hand': 'Right_Hand',
        
        # Left leg (REQUIRED: left_foot)
        'left_upper_leg': 'Left_UpperLeg',
        'left_lower_leg': 'Left_LowerLeg',
        'left_foot': 'Left_Foot',
        'left_toe': 'Left_Toe',
        
        # Right leg (REQUIRED: right_foot)
        'right_upper_leg': 'Right_UpperLeg',
        'right_lower_leg': 'Right_LowerLeg',
        'right_foot': 'Right_Foot',
        'right_toe': 'Right_Toe',
        
        # Optional: Fingers (if you want hand tracking)
        'left_carpus': 'Left_Carpus',
        'right_carpus': 'Right_Carpus',
    }
    
    # Minimum required body parts for basic retargeting
    REQUIRED_BODIES = [
        'Pelvis',
        'Head', 
        'Left_Hand',
        'Right_Hand',
        'Left_Foot',
        'Right_Foot',
    ]
    
    def __init__(self, port: int = 8001, verbose: bool = True):
        """
        Initialize the Xsens to GMR adapter.
        
        Args:
            port: UDP port for Xsens MVN streaming (default: 8001)
            verbose: Print debug information
        """
        self.device = XsensWrapper(port)
        self.verbose = verbose
        self.initialized = False

        # Store link names from Xsens
        self.xsens_link_names = []
        self.available_mappings = {}

        # Statistics
        self.frame_count = 0
        self.last_sample_counter = -1

        # Initial yaw normalization
        self.initial_yaw_captured = False
        self.initial_yaw_inv = None  # Inverse of initial pelvis yaw rotation
        
    def initialize(self) -> bool:
        """
        Initialize the Xsens device and discover available body parts.
        
        Returns:
            True if initialization successful, False otherwise
        """
        if self.verbose:
            print("🔌 Initializing Xsens MVN Device...")
            
        if not self.device.init():
            print("❌ Failed to initialize Xsens MVN device")
            return False
        
        # Get available link names
        self.xsens_link_names = self.device.get_link_names()
        
        if self.verbose:
            print(f"✅ Device initialized with {len(self.xsens_link_names)} links")
            print("\n📋 Available Xsens Links:")
            for i, name in enumerate(self.xsens_link_names):
                print(f"  [{i:2d}] {name}")
        
        # Build mapping of available bodies
        self._build_available_mappings()
        
        # Check if we have minimum required bodies
        missing = self._check_required_bodies()
        if missing:
            print(f"\n⚠️  WARNING: Missing required body parts: {missing}")
            print("    Retargeting may not work properly!")
            if not self.verbose:
                return False
        
        self.initialized = True
        return True
    
    def _build_available_mappings(self):
        """Build dictionary of available body mappings."""
        self.available_mappings = {}
        
        for xsens_name, gmr_name in self.XSENS_TO_GMR_MAPPING.items():
            if xsens_name in self.xsens_link_names:
                self.available_mappings[gmr_name] = xsens_name
        
        if self.verbose:
            print(f"\n✅ Found {len(self.available_mappings)} mapped bodies:")
            for gmr_name, xsens_name in self.available_mappings.items():
                is_required = gmr_name in self.REQUIRED_BODIES
                marker = "⭐" if is_required else "  "
                print(f"  {marker} {gmr_name:20s} <- {xsens_name}")
    
    def _check_required_bodies(self) -> List[str]:
        """
        Check which required bodies are missing.
        
        Returns:
            List of missing required body names
        """
        missing = []
        for required in self.REQUIRED_BODIES:
            if required not in self.available_mappings:
                missing.append(required)
        return missing
    
    def start(self):
        """Start Xsens data streaming."""
        if not self.initialized:
            raise RuntimeError("Device not initialized. Call initialize() first.")
        
        if self.verbose:
            print("\n🚀 Starting Xsens streaming...")
        self.device.start()
    
    def stop(self):
        """Stop Xsens data streaming."""
        if self.verbose:
            print("\n🛑 Stopping Xsens streaming...")
        self.device.stop()
    
    def get_human_frame(self) -> Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]]:
        """
        Get current frame data in GMR format.
        
        Returns:
            Dictionary mapping body names to (position, orientation) tuples:
            {
                "Body_Name": ([x, y, z], [qw, qx, qy, qz]),
                ...
            }
            Returns None if data cannot be retrieved.
        """
        if not self.initialized:
            raise RuntimeError("Device not initialized. Call initialize() first.")
        
        # Check if we have new data
        current_counter = int(self.device.get_sample_counter())
        if current_counter == self.last_sample_counter:
            # No new data, return None
            return None
        
        self.last_sample_counter = current_counter
        self.frame_count += 1
        
        human_frame = {}
        
        # Iterate through available mappings and get data
        for gmr_name, xsens_name in self.available_mappings.items():
            try:
                # Get position (in meters)
                pos = self.device.get_link_position(xsens_name)
                
                # Get orientation (as quaternion)
                ori = self.device.get_link_orientation(xsens_name)
                
                # Ensure position is numpy array
                if not isinstance(pos, np.ndarray):
                    pos = np.array(pos, dtype=np.float64)
                
                # Ensure orientation is numpy array and in correct format
                if not isinstance(ori, np.ndarray):
                    ori = np.array(ori, dtype=np.float64)
                
                # Check if orientation is in scalar-last format [x, y, z, w]
                # and convert to scalar-first [w, x, y, z] if needed
                # Xsens typically provides [w, x, y, z] already, but we verify
                ori_scalar_first = self._ensure_scalar_first_quaternion(ori)

                # Add to human frame
                human_frame[gmr_name] = (pos, ori_scalar_first)
                
            except Exception as e:
                if self.verbose:
                    print(f"⚠️  Warning: Could not get data for {gmr_name} ({xsens_name}): {e}")
                continue
        
        # Check if we got minimum required bodies
        if not self._validate_frame(human_frame):
            if self.verbose:
                print("⚠️  Warning: Frame missing required bodies")
            return None

        # Apply initial yaw normalization
        human_frame = self._apply_yaw_normalization(human_frame)

        return human_frame

    def _apply_yaw_normalization(self, human_frame: Dict) -> Dict:
        """
        Normalize the initial pelvis yaw so all recordings start facing forward.

        On the first frame, captures the pelvis yaw and computes its inverse.
        All subsequent frames have this inverse yaw applied to positions and orientations.
        """
        if 'Pelvis' not in human_frame:
            return human_frame

        _, pelvis_quat = human_frame['Pelvis']

        # Capture initial yaw on first frame
        if not self.initial_yaw_captured:
            # Extract yaw from pelvis quaternion
            pelvis_rot = R.from_quat(pelvis_quat, scalar_first=True)
            # Get euler angles (ZYX convention: yaw, pitch, roll)
            euler = pelvis_rot.as_euler('ZYX')
            yaw = euler[0]

            # Create yaw-only rotation and its inverse
            self.initial_yaw_inv = R.from_euler('Z', -yaw)
            self.initial_yaw_captured = True

            if self.verbose:
                print(f"📐 Initial pelvis yaw: {np.degrees(yaw):.1f}° - normalizing to 0°")

        # Apply inverse yaw to all body parts
        normalized_frame = {}
        for body_name, (pos, quat) in human_frame.items():
            # Rotate position around world Z-axis
            new_pos = self.initial_yaw_inv.apply(pos)

            # Rotate orientation (pre-multiply: new_rot = yaw_inv * body_rot)
            body_rot = R.from_quat(quat, scalar_first=True)
            new_rot = self.initial_yaw_inv * body_rot
            new_quat = new_rot.as_quat(scalar_first=True)

            normalized_frame[body_name] = (new_pos, new_quat)

        return normalized_frame

    def reset_yaw_normalization(self):
        """Reset yaw normalization to recapture on next frame."""
        self.initial_yaw_captured = False
        self.initial_yaw_inv = None
        if self.verbose:
            print("🔄 Yaw normalization reset")
    
    def _ensure_scalar_first_quaternion(self, quat: np.ndarray) -> np.ndarray:
        """
        Ensure quaternion is in scalar-first format [w, x, y, z].

        Args:
            quat: Quaternion array (either [w,x,y,z] or [x,y,z,w])

        Returns:
            Quaternion in [w, x, y, z] format
        """
        if len(quat) != 4:
            raise ValueError(f"Invalid quaternion length: {len(quat)}")

        # Normalize quaternion
        quat = quat / np.linalg.norm(quat)

        return quat  # assume scalar-first
    
    def _validate_frame(self, human_frame: Dict) -> bool:
        """
        Validate that frame contains minimum required bodies.
        
        Args:
            human_frame: Dictionary of body data
            
        Returns:
            True if frame has all required bodies, False otherwise
        """
        for required in self.REQUIRED_BODIES:
            if required not in human_frame:
                return False
        return True
    
    def get_frame_info(self) -> Dict:
        """
        Get information about the current frame.
        
        Returns:
            Dictionary with frame metadata
        """
        return {
            'sample_counter': int(self.device.get_sample_counter()),
            'frame_time': int(self.device.get_frame_time()),
            'total_frames': self.frame_count,
        }
    
    def estimate_missing_bodies(self, human_frame: Dict) -> Dict:
        """
        Estimate positions for missing required bodies based on available data.
        This is a simple fallback for when some tracking is lost.
        
        Args:
            human_frame: Current human frame (possibly incomplete)
            
        Returns:
            Updated human frame with estimated bodies
        """
        # Only estimate if we have pelvis as reference
        if 'Pelvis' not in human_frame:
            return human_frame
        
        pelvis_pos, pelvis_quat = human_frame['Pelvis']
        
        # Estimate Head if missing (assume ~0.6m above pelvis)
        if 'Head' not in human_frame:
            head_pos = pelvis_pos + np.array([0.0, 0.0, 0.6])
            human_frame['Head'] = (head_pos, pelvis_quat)
        
        # Estimate hands if missing (assume at shoulder height, to the sides)
        if 'Left_Hand' not in human_frame:
            left_hand_pos = pelvis_pos + np.array([-0.5, 0.0, 0.4])
            human_frame['Left_Hand'] = (left_hand_pos, pelvis_quat)
        
        if 'Right_Hand' not in human_frame:
            right_hand_pos = pelvis_pos + np.array([0.5, 0.0, 0.4])
            human_frame['Right_Hand'] = (right_hand_pos, pelvis_quat)
        
        # Estimate feet if missing (assume ~0.1m above ground, hip-width apart)
        if 'Left_Foot' not in human_frame:
            left_foot_pos = pelvis_pos + np.array([-0.15, 0.0, -0.9])
            human_frame['Left_Foot'] = (left_foot_pos, pelvis_quat)
        
        if 'Right_Foot' not in human_frame:
            right_foot_pos = pelvis_pos + np.array([0.15, 0.0, -0.9])
            human_frame['Right_Foot'] = (right_foot_pos, pelvis_quat)
        
        return human_frame
    
    def print_frame_summary(self, human_frame: Dict):
        """
        Print a summary of the current frame data.
        
        Args:
            human_frame: Current human frame data
        """
        print("\n" + "="*70)
        print(f"📊 Frame #{self.frame_count} Summary")
        print("="*70)
        
        for body_name, (pos, quat) in sorted(human_frame.items()):
            is_required = body_name in self.REQUIRED_BODIES
            marker = "⭐" if is_required else "  "
            print(f"{marker} {body_name:20s}: pos=[{pos[0]:+7.3f}, {pos[1]:+7.3f}, {pos[2]:+7.3f}] "
                  f"quat=[{quat[0]:+6.3f}, {quat[1]:+6.3f}, {quat[2]:+6.3f}, {quat[3]:+6.3f}]")
        
        print("="*70)


def test_xsens_to_gmr(port: int = 8001, duration: float = 5.0):
    """
    Test function to verify XsensToGMR adapter.
    
    Args:
        port: UDP port for Xsens streaming
        duration: How long to run test (seconds)
    """
    import time
    
    print("="*70)
    print("🧪 Testing XsensToGMR Adapter")
    print("="*70)
    
    # Initialize adapter
    adapter = XsensToGMR(port=port, verbose=True)
    
    if not adapter.initialize():
        print("❌ Failed to initialize adapter")
        return
    
    # Start streaming
    adapter.start()
    time.sleep(1.0)  # Wait for streaming to stabilize
    
    print(f"\n🎬 Collecting data for {duration} seconds...")
    print("Press Ctrl+C to stop early\n")
    
    start_time = time.time()
    frame_count = 0
    last_print = start_time
    
    try:
        while time.time() - start_time < duration:
            # Get human frame
            human_frame = adapter.get_human_frame()
            
            if human_frame is not None:
                frame_count += 1
                
                # Print summary every 2 seconds
                if time.time() - last_print >= 2.0:
                    adapter.print_frame_summary(human_frame)
                    
                    info = adapter.get_frame_info()
                    print(f"\n📈 Stats: {frame_count} frames in {time.time() - start_time:.1f}s "
                          f"({frame_count / (time.time() - start_time):.1f} FPS)")
                    
                    last_print = time.time()
            
            # Small sleep to avoid busy waiting
            time.sleep(0.001)
    
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
    
    finally:
        # Stop streaming
        adapter.stop()
        
        elapsed = time.time() - start_time
        print("\n" + "="*70)
        print(f"✅ Test completed: {frame_count} frames in {elapsed:.2f}s "
              f"({frame_count / elapsed:.1f} FPS)")
        print("="*70)


if __name__ == '__main__':
    import sys
    
    # Parse command line arguments
    port = 9763
    duration = 5.0
    
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            print(f"Invalid port '{sys.argv[1]}', using default {port}")
    
    if len(sys.argv) > 2:
        try:
            duration = float(sys.argv[2])
        except ValueError:
            print(f"Invalid duration '{sys.argv[2]}', using default {duration}s")
    
    # Run test
    test_xsens_to_gmr(port, duration)
