## Model for d2q9 SRT BKG-LBM
#     Density - performs streaming operation for us
#

# Add the particle distribution functions as model Densities:
AddDensity( name="f0", dx= 0, dy= 0, group="f")
AddDensity( name="f1", dx= 1, dy= 0, group="f")
AddDensity( name="f2", dx= 0, dy= 1, group="f")
AddDensity( name="f3", dx=-1, dy= 0, group="f")
AddDensity( name="f4", dx= 0, dy=-1, group="f")
AddDensity( name="f5", dx= 1, dy= 1, group="f")
AddDensity( name="f6", dx=-1, dy= 1, group="f")
AddDensity( name="f7", dx=-1, dy=-1, group="f")
AddDensity( name="f8", dx= 1, dy=-1, group="f")

#AddField(name="f[1]", dx=1);
AddField(name="f0", dx=-1, dy=0);
AddField(name="f1", dx=-1, dy=0);
AddField(name="f2", dx=-1, dy=0);
AddField(name="f3", dx=-1, dy=0);
AddField(name="f4", dx=-1, dy=0);
AddField(name="f5", dx=-1, dy=0);
AddField(name="f6", dx=-1, dy=0);
AddField(name="f7", dx=-1, dy=0);
AddField(name="f8", dx=-1, dy=0);

# Add the quantities we wish to be exported
#    These quantities must be defined by a function in Dynamics.c
AddQuantity( name="U",unit="m/s", vector=TRUE )
AddQuantity( name="Rho",unit="kg/m3" )


densities<-paste0("fout",seq(0,8))
for (den in densities){
  AddQuantity( name=den, unit=1)
}

# Add the settings which describes system constants defined in a .xml file
AddSetting( name="omega", comment='inverse of relaxation time')
AddSetting( name="nu", omega='1.0/(3*nu+0.5)', default=0.16666666, comment='viscosity')
AddSetting( name="Velocity",default=0, comment='inlet/outlet/init velocity', zonal=TRUE)
AddSetting( name="Velocity_x",default=0, comment='inlet/outlet/init velocity in x', zonal=TRUE )
AddSetting( name="Velocity_x_init",default=0, comment='inlet/outlet/init velocity in x', zonal=TRUE )
AddSetting( name="Velocity_y_init",default=0, comment='inlet/outlet/init velocity in x', zonal=TRUE )
AddSetting( name="Velocity_x_BC",default=0, comment='inlet/outlet/init velocity in x', zonal=TRUE )
AddSetting( name="Velocity_y",default=0, comment='inlet/outlet/init velocity in y', zonal=TRUE )
AddSetting( name="GravitationX",default=0, comment='body/external acceleration', zonal=TRUE)
AddSetting( name="GravitationY",default=0, comment='body/external acceleration', zonal=TRUE)
AddSetting( name="disturbance_ax",default=0, comment='initial disturbance', zonal=TRUE)
AddSetting( name="disturbance_ay",default=0, comment='initial disturbance', zonal=TRUE)
AddSetting( name="disturbance_b",default=0, comment='initial disturbance', zonal=TRUE)
AddSetting( name="disturbance_c",default=0, comment='initial disturbance', zonal=TRUE)
AddSetting( name="position_init_shockwave",default=0, comment='position_init_shockwave', zonal=TRUE)
AddSetting( name="b_pressure",default=3, comment='b_pressure', zonal=TRUE)
AddSetting( name="eps_pressure",default=0.001, comment='eps_pressure', zonal=TRUE)
AddSetting( name="disturbance_norm",default=0, comment='initial disturbance', zonal=TRUE)
AddSetting( name="DisturbanceWavenumber",default=0, comment='initial wave number', zonal=TRUE)
AddSetting( name="Density",default=1, comment='Density')
AddSetting( name="tmp1",default=1, comment='temporary variable')
AddSetting( name="tmp2",default=1, comment='temporary variable')
AddSetting( name="YY",default=1, comment='temporary variable')
AddSetting( name="XX",default=1, comment='temporary variable')
AddSetting( name="Ax",default=1, comment='temporary variable')
AddSetting( name="Ay",default=1, comment='temporary variable')
AddSetting( name="InitField",default=3, comment='Specify Initialization')
AddSetting( name="xc",default=100, comment='Specify Initialization')
AddSetting( name="yc",default=100, comment='Specify Initialization')
AddSetting( name="b",default=0.1, comment='Specify Initialization')

AddSetting(name="S2", default="0", comment='MRT Sx')
AddSetting(name="S3", default="0", comment='MRT Sx')
AddSetting(name="S4", default="0", comment='MRT Sx')

AddGlobal(name="Drag", comment='Force exerted on body in X-direction', unit="N")
AddGlobal(name="Lift", comment='Force exerted on body in Z-direction', unit="N")
#AddGlobal(name="Drag_new", comment='Force exerted on body in X-direction', unit="m4/s2")
#AddGlobal(name="Lift_new", comment='Force exerted on body in Z-direction', unit="m4/s2")

AddNodeType(name="EPressure", group="BOUNDARY")
AddNodeType(name="EEQPressure", group="BOUNDARY")
AddNodeType(name="EPressure_nbc1", group="BOUNDARY")
AddNodeType(name="WPressure", group="BOUNDARY")
AddNodeType(name="WVelocity", group="BOUNDARY")
AddNodeType(name="EVelocity", group="BOUNDARY")
AddNodeType(name="Solid", group="BOUNDARY")
AddNodeType(name="Wall", group="BOUNDARY")
AddNodeType(name="BGK", group="COLLISION")
AddNodeType(name="MRT", group="COLLISION")
AddNodeType(name="Body", group="BODY")
