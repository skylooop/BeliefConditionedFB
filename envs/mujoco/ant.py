import gymnasium as gym
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.envs.mujoco.ant_v5 import AntEnv as AntEnvOrig
import mujoco

import re
import os
import numpy as np
import tempfile

class AntEnv(AntEnvOrig):
    def __init__(self, default_ind=0, num_envs=20, radius=4.0, viscosity=0.05, basepath=None, **kwargs):
        self.num_envs = num_envs
        self.default_params = {'limbs': [.2, .2, .2, .2], 'wind': [0, 0, 0], 'viscosity': 0.0}
        self.default_ind = default_ind #kwargs.pop('default_ind')
        self.env_configs = []
        self.current_temp_dir = None  # Track active temporary directory

        # Create environment configurations
        for i in range(num_envs):
            angle = i * (2*np.pi/num_envs)
            self.env_configs.append({
                'limbs': [.2, .2, .2, .2],
                'wind': [radius*np.cos(angle), radius*np.sin(angle), 0],
                'viscosity': viscosity
            })
        self.env_configs.append(self.default_params)

        # Load base XML
        self.basepath = basepath or os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(self.basepath, "ant.xml")) as f:
            self.base_xml = f.read()

        # Initial setup with default config
        self._active_xml = self._modify_xml(self.default_ind)
        self._temp_file = self._create_temp_xml()
        super().__init__(xml_file=self._temp_file.name, **kwargs)

    def _modify_xml(self, ind):
        params = {**self.default_params, **self.env_configs[ind]}
        wx, wy, wz = params['wind']
        viscosity = params['viscosity']
        limbs = params['limbs']

        xml = re.sub(
            r'<option integrator="RK4" timestep="0.01"/>',
            f'<option integrator="RK4" timestep="0.01" wind="{wx} {wy} {wz}" viscosity="{viscosity}"/>',
            self.base_xml
        )
        
        replacements = {
            " 0.2 0.2": f" {limbs[0]} {limbs[0]}",
            "-0.2 0.2": f"-{limbs[1]} {limbs[1]}",
            " 0.2 -0.2": f" {limbs[2]} -{limbs[2]}",
            "-0.2 -0.2": f"-{limbs[3]} -{limbs[3]}"
        }
        
        for pattern, replacement in replacements.items():
            xml = xml.replace(pattern, replacement)
        
        return xml

    def _create_temp_xml(self):
        # Create persistent temporary file
        temp_file = tempfile.NamedTemporaryFile(
            mode='w', 
            suffix='.xml', 
            delete=False,  # Keep file until explicitly deleted
            prefix='ant_',
            dir=self.current_temp_dir
        )
        temp_file.write(self._active_xml)
        temp_file.close()
        return temp_file

    def reset(self, *, seed=None, options=None, env_id=None, same=False):
        if not same and env_id is not None:
            self.ind = env_id or self.default_ind
            self._active_xml = self._modify_xml(self.ind)
            
            # Cleanup previous temp files
            if hasattr(self, '_temp_file'):
                os.unlink(self._temp_file.name)
            
            # Create new temp file
            self._temp_file = self._create_temp_xml()
            
            # Reinitialize simulation
            self.xml_file = self._temp_file.name
            self._initialize_simulation()

        return super().reset(seed=seed, options=options)

    def close(self):
        # Cleanup temporary files on environment close
        if hasattr(self, '_temp_file'):
            os.unlink(self._temp_file.name)
        super().close()