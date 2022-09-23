#pip install pywin32
import win32api, win32con

def click(x,y):
    win32api.SetCursorPos((x,y))
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN,x,y,0,0)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP,x,y,0,0)

click(10,10)


pos = win32api.GetCursorPos()
print(pos)

pos = (200, 200)
win32api.SetCursorPos(pos)

import pywinauto

img = app.capture_as_image()
img.save('screenshot.png')

rectangle()#This method returns a RECT structure, Which has attributes - top, left, right, bottom. and has methods width() and height(). See win32structures.RECT for more information.
'''
classpywinauto.base_wrapper.BaseWrapper(element_info, active_backend)
Abstract wrapper for elements.

All other wrappers are derived from this.

can_be_label= False
capture_as_image(rect=None)
Return a PIL image of the control.

See PIL documentation to know what you can do with the resulting image.

children(**kwargs)
Return the children of this element as a list

It returns a list of BaseWrapper (or subclass) instances. An empty list is returned if there are no children.

class_name()
Return the class name of the elenemt

click_input(button='left', coords=(None, None), button_down=True, button_up=True, double=False, wheel_dist=0, use_log=True, pressed='', absolute=False, key_down=True, key_up=True)
Click at the specified coordinates

button The mouse button to click. One of ‘left’, ‘right’, ‘middle’ or ‘x’ (Default: ‘left’, ‘move’ is a special case)
coords The coordinates to click at.(Default: the center of the control)
double Whether to perform a double click or not (Default: False)
wheel_dist The distance to move the mouse wheel (default: 0)
NOTES:
This is different from click method in that it requires the control to be visible on the screen but performs a more realistic ‘click’ simulation.

This method is also vulnerable if the mouse is moved by the user as that could easily move the mouse off the control before the click_input has finished.

client_to_screen(client_point)
Maps point from client to screen coordinates

control_count()
Return the number of children of this control

control_id()
Return the ID of the element

Only controls have a valid ID - dialogs usually have no ID assigned.

The ID usually identified the control in the window - but there can be duplicate ID’s for example lables in a dialog may have duplicate ID’s.

descendants(**kwargs)
Return the descendants of this element as a list

It returns a list of BaseWrapper (or subclass) instances. An empty list is returned if there are no descendants.

double_click_input(button='left', coords=(None, None))
Double click at the specified coordinates

drag_mouse_input(dst=(0, 0), src=None, button='left', pressed='', absolute=True)
Click on src, drag it and drop on dst

dst is a destination wrapper object or just coordinates.
src is a source wrapper object or coordinates. If src is None the self is used as a source object.
button is a mouse button to hold during the drag. It can be “left”, “right”, “middle” or “x”
pressed is a key on the keyboard to press during the drag.
absolute specifies whether to use absolute coordinates for the mouse pointer locations
draw_outline(colour='green', thickness=2, fill=<MagicMock name='mock.win32defines.BS_NULL' id='140124673757368'>, rect=None)
Draw an outline around the window.

colour can be either an integer or one of ‘red’, ‘green’, ‘blue’ (default ‘green’)
thickness thickness of rectangle (default 2)
fill how to fill in the rectangle (default BS_NULL)
rect the coordinates of the rectangle to draw (defaults to the rectangle of the control)
element_info
Read-only property to get ElementInfo object

friendly_class_name()
Return the friendly class name for the control

This differs from the class of the control in some cases. class_name() is the actual ‘Registered’ element class of the control while friendly_class_name() is hopefully something that will make more sense to the user.

For example Checkboxes are implemented as Buttons - so the class of a CheckBox is “Button” - but the friendly class is “CheckBox”

friendlyclassname= None
from_point(x, y)
Get wrapper object for element at specified screen coordinates (x, y)

get_properties()
Return the properties of the control as a dictionary.

has_title= True
is_child(parent)
Return True if this element is a child of ‘parent’.

An element is a child of another element when it is a direct of the other element. An element is a direct descendant of a given element if the parent element is the the chain of parent elements for the child element.

is_dialog()
Return True if the control is a top level window

is_enabled()
Whether the element is enabled or not

Checks that both the top level parent (probably dialog) that owns this element and the element itself are both enabled.

If you want to wait for an element to become enabled (or wait for it to become disabled) use Application.wait('visible') or Application.wait_not('visible').

If you want to raise an exception immediately if an element is not enabled then you can use the BaseWrapper.verify_enabled(). BaseWrapper.VerifyReady() raises if the window is not both visible and enabled.

is_visible()
Whether the element is visible or not

Checks that both the top level parent (probably dialog) that owns this element and the element itself are both visible.

If you want to wait for an element to become visible (or wait for it to become hidden) use Application.wait('visible') or Application.wait_not('visible').

If you want to raise an exception immediately if an element is not visible then you can use the BaseWrapper.verify_visible(). BaseWrapper.verify_actionable() raises if the element is not both visible and enabled.

iter_children(**kwargs)
Iterate over the children of this element

It returns a generator of BaseWrapper (or subclass) instances.

iter_descendants(**kwargs)
Iterate over the descendants of this element

It returns a generator of BaseWrapper (or subclass) instances.

move_mouse_input(coords=(0, 0), pressed='', absolute=True)
Move the mouse

parent()
Return the parent of this element

Note that the parent of a control is not necesarily a dialog or other main window. A group box may be the parent of some radio buttons for example.

To get the main (or top level) window then use BaseWrapper.top_level_parent().

press_mouse_input(button='left', coords=(None, None), pressed='', absolute=True, key_down=True, key_up=True)
Press a mouse button using SendInput

process_id()
Return the ID of process that owns this window

rectangle()
Return the rectangle of element

The rectangle() is the rectangle of the element on the screen. Coordinates are given from the top left of the screen.

This method returns a RECT structure, Which has attributes - top, left, right, bottom. and has methods width() and height(). See win32structures.RECT for more information.

release_mouse_input(button='left', coords=(None, None), pressed='', absolute=True, key_down=True, key_up=True)
Release the mouse button

right_click_input(coords=(None, None))
Right click at the specified coords

root()
Return wrapper for root element (desktop)

set_focus()
Set the focus to this element

texts()
Return the text for each item of this control

It is a list of strings for the control. It is frequently overridden to extract all strings from a control with multiple items.

It is always a list with one or more strings:

The first element is the window text of the control
Subsequent elements contain the text of any items of the control (e.g. items in a listbox/combobox, tabs in a tabcontrol)
top_from_point(x, y)
Get wrapper object for top level element at specified screen coordinates (x, y)

top_level_parent()
Return the top level window of this control

The TopLevel parent is different from the parent in that the parent is the element that owns this element - but it may not be a dialog/main window. For example most Comboboxes have an Edit. The ComboBox is the parent of the Edit control.

This will always return a valid window element (if the control has no top level parent then the control itself is returned - as it is a top level window already!)

type_keys(keys, pause=None, with_spaces=False, with_tabs=False, with_newlines=False, turn_off_numlock=True, set_foreground=True, vk_packet=True)
Type keys to the element using keyboard.send_keys

This uses the re-written keyboard python module where you can find documentation on what to use for the keys.

verify_actionable()
Verify that the element is both visible and enabled

Raise either ElementNotEnalbed or ElementNotVisible if not enabled or visible respectively.

verify_enabled()
Verify that the element is enabled

Check first if the element’s parent is enabled (skip if no parent), then check if element itself is enabled.

verify_visible()
Verify that the element is visible

Check first if the element’s parent is visible. (skip if no parent), then check if element itself is visible.

wait_for_idle()
Backend specific function to wait for idle state of a thread or a window

was_maximized()
Indicate whether the window was maximized before minimizing or not

wheel_mouse_input(coords=(None, None), wheel_dist=1, pressed='')
Do mouse wheel

window_text()
Window text of the element

Quite a few contorls have other text that is visible, for example Edit controls usually have an empty string for window_text but still have text displayed in the edit window.

windowclasses= []
writable_props
Build the list of the default properties to be written.

Derived classes may override or extend this list depending on how much control they need.
'''