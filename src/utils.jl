"""
    currentvalue(s::ParameterSchedulers.Stateful)

Return the current value of the scheduler `s`.

!!! warning
    Really, this should be have been implemented somewhere in
    `ParameterSchedulers.jl`.
"""
currentvalue(s::ParameterSchedulers.Stateful) = s.schedule(s.state)