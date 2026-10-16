/// @file library.hpp
/// @ingroup group_comm
#pragma once

namespace numkit {

/// @addtogroup group_comm
class PluginContext;

class CommLibrary
{
public:
    static void install(PluginContext &engine);
};


/// @}
} // namespace numkit
