import matplotlib.pyplot as plt
import matplotlib as mpl

def update_label(old_label, exponent_text):
    if exponent_text == "":
        return old_label
    
    try:
        units = old_label[old_label.index("[") + 1:old_label.rindex("]")]
    except ValueError:
        units = ""
    label = old_label.replace("[{}]".format(units), "")
    
    exponent_text = exponent_text.replace("\\times", "")
    
    return "{} [{} {}]".format(label, exponent_text, units)
    
def format_label_string_with_exponent(ax, axis='both'):  
    """ Format the label string with the exponent from the ScalarFormatter """
    ax.ticklabel_format(axis=axis, style='sci')

    axes_instances = []
    if axis in ['x', 'both']:
        axes_instances.append(ax.xaxis)
    if axis in ['y', 'both']:
        axes_instances.append(ax.yaxis)
    
    for ax in axes_instances:
        ax.major.formatter._useMathText = True
        plt.draw() # Update the text
        exponent_text = ax.get_offset_text().get_text()
        label = ax.get_label().get_text()
        ax.offsetText.set_visible(False)
        ax.set_label_text(update_label(label, exponent_text))