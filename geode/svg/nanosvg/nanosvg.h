//
// Copyright (c) 2009 Mikko Mononen memon@inside.org
//
// This software is provided 'as-is', without any express or implied
// warranty.  In no event will the authors be held liable for any damages
// arising from the use of this software.
// Permission is granted to anyone to use this software for any purpose,
// including commercial applications, and to alter it and redistribute it
// freely, subject to the following restrictions:
// 1. The origin of this software must not be misrepresented; you must not
//    claim that you wrote the original software. If you use this software
//    in a product, an acknowledgment in the product documentation would be
//    appreciated but is not required.
// 2. Altered source versions must be plainly marked as such, and must not be
//    misrepresented as being the original software.
// 3. This notice may not be removed or altered from any source distribution.
//

// Version 1.0 - Initial version
// Version 1.1 - Fixed path parsing, implemented curves, implemented circle.
// This source has been altered from Version 1.1: added bezier curve output.

#ifndef NANOSVG_H
#define NANOSVG_H

#include <geode/utility/config.h> // for GEODE_EXPORT

#ifdef __cplusplus
extern "C" {
#endif

/* Example Usage:
  // Load
  struct SVGPath* plist;
  plist = svgParseFromFile("test.svg");
  // Use...
  for (SVGPath* it = plist; it; it = it->next)
    ...
  // Delete
  svgDelete(plist);
*/

struct SVGInfo
{
  char x_unit[2], y_unit[2], width_unit[2], height_unit[2];
  int x, y, width, height;

  char viewbox_units[4][2];
  int viewbox[4];

  char preserveAspectRatio[9];
};

struct SVGPath
{
  // polyline
  float* pts;
  int npts;

  // added: bezier segments
  float* bezpts;
  int nbezpts;

  unsigned int elementIndex; //can be used to differentiate multiple subpaths from same path. Will monotonicaly decrease while traversing next
  unsigned int fillColor;
  unsigned int strokeColor;
  float strokeWidth;
  int hasFill;
  char fillRule; // 1 for nonzero (default), 2 for evenodd
  char hasStroke;
  int closed;
  char *CSSclass;
  struct SVGPath* next;
};

// Parses SVG file from a file, returns linked list of paths.
GEODE_CORE_EXPORT struct SVGPath* svgParseFromFile(const char* filename, struct SVGInfo *);

// Parses SVG file from a null terminated string, returns linked list of paths.
GEODE_CORE_EXPORT struct SVGPath* svgParse(char* input, struct SVGInfo *);

// Deletes list of paths.
GEODE_CORE_EXPORT void svgDelete(struct SVGPath* plist);

#ifdef __cplusplus
};
#endif

#endif // NANOSVG_H
