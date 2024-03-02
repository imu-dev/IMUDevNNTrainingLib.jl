const _RED = (255, 50, 100)
const _GREEN = (60, 171, 71)

function info_panel(text)
    return Panel(text; fit=true,
                 subtitle="{$_RED}info{/$_RED}",
                 subtitle_style="bold")
end

function summary_panel(text; title="Training")
    return Panel(text; fit=true,
                 subtitle="{$_GREEN}summary{/$_GREEN}",
                 subtitle_style="bold",
                 title)
end

"""
    start_info(loader::Flux.DataLoader)

Print basic information about the `loader` and the data it holds.

!!! tip
    This method is intended to be called at the start of training.
"""
function start_info(loader::Flux.DataLoader)
    info = basic_info_as_string(loader)

    p = Panel(info_panel("STARTING TRAINING"),
              info_panel(info); fit=true)
    println(p)
    return nothing
end

"""
    echo_epoch(id)

Print the current epoch number.
"""
function echo_epoch(id)
    println(info_panel("EPOCH $(digitsep(id; seperator='_'))"))
    return nothing
end

"""
    echo_summary(; epoch, avg_loss, elapsed, title="Training")

Print a summary of the training process.
"""
function echo_summary(; epoch, avg_loss, elapsed, title="Training")
    p = summary_panel("""Epoch $epoch. Loss: $avg_loss
                  Elapsed: $(round(elapsed, Second))"""; title)
    println(p)
    return nothing
end
