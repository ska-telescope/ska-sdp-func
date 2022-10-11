*******
Logging
*******

Errors can be reported by calling :any:`SDP_LOG_ERROR`, with a
printf-style format string and arguments.
This automatically adds the required fields needed to comply with the
SKA logging standard.

Include the header *"ska-sdp-func/utility/sdp_logging.h"* to use
these functions.

.. doxygendefine:: SDP_LOG_CRITICAL

.. doxygendefine:: SDP_LOG_ERROR

.. doxygendefine:: SDP_LOG_WARNING

.. doxygendefine:: SDP_LOG_INFO

.. doxygendefine:: SDP_LOG_DEBUG


Log message filtering
=====================
Log messages can be suppressed below a specified level by setting the
value of the environment variable `SKA_SDP_FUNC_LOG_LEVEL`. The value should
be a string ("DEBUG", "INFO", "WARNING", "ERROR" or "CRITICAL") to specify the
minimum log message level that will be displayed in the terminal.
