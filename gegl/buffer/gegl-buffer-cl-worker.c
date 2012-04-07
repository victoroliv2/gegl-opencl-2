#include "config.h"
#include <glib.h>

#include "gegl.h"
#include "gegl/gegl-debug.h"
#include "gegl-utils.h"
#include "gegl-types-internal.h"
#include "gegl-buffer-types.h"
#include "gegl-buffer.h"
#include "gegl-buffer-private.h"

#include "gegl-buffer-cl-worker.h"

/* Worker CPU threads to load or store data in the OpenCL device */

typedef struct ThreadData
{
  gint           tid;
  GeglBuffer    *buffer;
  GeglRectangle  roi;
  gpointer       buf;
  gboolean       write;
} ThreadData;

#define GEGL_CL_WORKER_THREADS 4

static GThreadPool *pool = NULL;
static GMutex *mutex = NULL;
static GCond  *cond  = NULL;
static gint    remaining_tasks = 0;

static void worker (gpointer data,
                    gpointer foo)
{
  ThreadData *td = data;
  GeglBuffer    *buffer = td->buffer;
  GeglRectangle  roi    = td->roi;
  gpointer       buf    = td->buf;
  gboolean       write  = td->write;

  if (GEGL_IS_BUFFER (buffer))
    {
      if (write)
        {
          gegl_buffer_set (buffer, &roi, 0, buffer->soft_format, buf, GEGL_AUTO_ROWSTRIDE);
        }
      else
        {
          gegl_buffer_get_unlocked (buffer, 1.0, &roi, buffer->soft_format, buf, GEGL_AUTO_ROWSTRIDE);
        }
    }

  g_mutex_lock (mutex);
  remaining_tasks --;
  if (remaining_tasks == 0)
    {
      g_cond_signal (cond);
    }
  g_mutex_unlock (mutex);
}

void
gegl_buffer_cl_worker_transf (GeglBuffer *buffer, gpointer data, size_t pixel_size, GeglRectangle roi, gboolean write)
{
  int tid;
  ThreadData tdata[GEGL_CL_WORKER_THREADS];
  int split = roi.height / GEGL_CL_WORKER_THREADS;
  size_t offset = 0;

  if (pool == NULL)
    {
      pool = g_thread_pool_new (worker, NULL, GEGL_CL_WORKER_THREADS, TRUE, NULL);
      mutex = g_mutex_new ();
      cond = g_cond_new ();
    }

  remaining_tasks += GEGL_CL_WORKER_THREADS;

  for (tid=0; tid < GEGL_CL_WORKER_THREADS; tid++)
    {
      GeglRectangle r = {roi.x, roi.y + split * tid, roi.width, split};
      if (tid == GEGL_CL_WORKER_THREADS - 1)
        r.height += roi.height % GEGL_CL_WORKER_THREADS;

      tdata[tid].tid    = tid;
      tdata[tid].roi    = r;
      tdata[tid].buffer = buffer;
      tdata[tid].buf    = (gchar*)data + offset;
      tdata[tid].write  = write;

      g_thread_pool_push (pool, &tdata[tid], NULL);

      offset += r.width * r.height * pixel_size;
    }

  g_mutex_lock (mutex);
  while (remaining_tasks!=0)
    g_cond_wait (cond, mutex);
  g_mutex_unlock (mutex);
}