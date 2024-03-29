#include <boost/python.hpp>
#include <face.h>
#include <string>


/* We only export a subset of the face functionality to python.
 * For more detail see include/face.h
 */
BOOST_PYTHON_MODULE_INIT(face)
{
    using namespace boost::python;
    using namespace face;
/*
    class_<Face>("Face", make_constructor(&Face::Face, default_call_policies(),
                (arg("x_len") = 640, arg("y_len") = 480), arg("x_weight") = 0.5,
                arg("y_weight") = 0.5, arg("dev") = "/dev/ttyACM0"))
*/
    class_<Face>("Face", "One must create an instance of this class "
                "to communicate with RoboFace.",
                init<int, int, float, float, const std::string &>(
                    (arg("x_len") = 640, arg("y_len") = 480, arg("x_weight") = 0.5,
                    arg("y_weight") = 0.5, arg("dev") = "/dev/ttyACM0"),
                    "x_len and y_len are the sizes(in pixels) of the webcam image\n"
                    "x_weight and y_weight determine how the moveHead(x,y) function"
                    " prioritizes the movement directions "
                    "- x_weight + y_weight must be equal to 1.0\n"
                    "dev is devicefile associated with the robots hardware"))
        .def("neutral", &Face::neutral, (arg("moveHead") = true),
                "neutral facial expression with optional head movement")
        .def("unsure", &Face::unsure, (arg("moveHead") = true),
                "unsure facial expression with optional head movement")
        .def("happy", &Face::happy, (arg("moveHead") = true),
                "happy facial expression with optional head movement")
        .def("angry", &Face::angry, (arg("moveHead") = true),
                "angry facial expression with optional head movement")
        .def("sad", &Face::sad, (arg("moveHead") = true),
                "sad facial expression with optional head movement")
        .def("moveHeadX", &Face::moveHeadX, (arg("x")),
                "move head horizontally")
        .def("moveHeadY", &Face::moveHeadY, (arg("y")),
                "move head vertically")
        .def("moveHead", &Face::moveHead, (arg("x"), arg("y")),
                "move head horizontally and vertically")
	.def("moveLips", &Face::moveLips, (arg("x")),
                "move lips in order to the given amplitude (0,10)")
        .def("setSpeedLips", &Face::setSpeedLips, (arg("v")),
                "Set Speed of Lips ")   
        .def("setSpeedHead", &Face::setSpeedHead, (arg("v")),
                "Set Speed of Head ")   
        .def("setSpeedEyes", &Face::setSpeedEyes, (arg("v")),
                "Set Speed of Eyes ")   
        .def("setSpeedBrows", &Face::setSpeedBrows, (arg("v")),
                "Set Speed of Brows ")        
        .def("setSpeedEars", &Face::setSpeedEars, (arg("v")),
                "Set Speed of Ears ")  
        .def("setSpeedAll", &Face::setSpeedAll, (arg("v")),
                "Set Speed for all motors")                
  
    ;
}






